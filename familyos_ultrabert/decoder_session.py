"""
DecoderSession - Lazy-Loaded Decoder for R5 Dream Exploration

Memory-efficient context manager that loads decoder only when needed.
Perfect for nightly P03 consolidation where decoder is used in R5 only.

Memory Model:
    ┌──────────────────────────────────────────────────────────────────┐
    │ MEMORY LIFECYCLE                                                  │
    │                                                                   │
    │ Daytime (R0-R4, R6-R8):                                          │
    │ ┌─────────────────────────┐                                      │
    │ │ Encoder + 12 Heads      │ 175 MB (INT8)                        │
    │ └─────────────────────────┘                                      │
    │                                                                   │
    │ R5 Start: decoder.__enter__()                                    │
    │ ┌─────────────────────────┐ ┌─────────────────────┐              │
    │ │ Encoder + 12 Heads      │ │ Decoder             │ +350 MB      │
    │ └─────────────────────────┘ └─────────────────────┘              │
    │                             Total: 525 MB                         │
    │                                                                   │
    │ R5 End: decoder.__exit__()                                       │
    │ ┌─────────────────────────┐                                      │
    │ │ Encoder + 12 Heads      │ 175 MB (decoder freed)               │
    │ └─────────────────────────┘                                      │
    └──────────────────────────────────────────────────────────────────┘

Usage:
    from familyos_ultrabert import Client
    from familyos_ultrabert.decoder_session import DecoderSession

    # Load encoder (always resident)
    client = Client()

    # R5: Load decoder temporarily
    with DecoderSession(quantization="int8") as decoder:
        for episode in episodes:
            encoder_output = client.encode(episode.text)
            suggestion = decoder.generate(encoder_output)

    # Decoder automatically unloaded, memory freed

Alternative Usage (via Client):
    with client.create_decoder_session() as decoder:
        suggestion = decoder.generate(encoder_output)
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# Type alias for quantization options
QuantizationType = Literal["fp32", "fp16", "int8"]


class DecoderSession:
    """
    Lazy-loaded decoder session for counterfactual generation.

    This context manager provides memory-efficient access to the GPT-2
    decoder. The decoder is loaded on __enter__ and unloaded on __exit__,
    making it perfect for R5 dream exploration phases.

    The decoder supports two backends:
    - "onnx": ONNX Runtime with automatic hardware selection (NPU/CUDA/CPU)
    - "pytorch": Full PyTorch model (more accurate, requires more memory)

    Args:
        version: Decoder version (default: "v3")
        quantization: Weight format - "fp32", "fp16", or "int8" (default: "int8")
        device: Device selection - "auto", "npu", "cuda", or "cpu" (default: "auto")
        backend: Inference backend - "onnx" or "pytorch" (default: "pytorch")
        max_batch_size: Maximum batch size for parallel generation (default: 16)
        cache_dir: Custom cache directory for weights (default: None uses default)

    Example:
        >>> with DecoderSession(backend="pytorch") as decoder:
        ...     result = decoder.generate(encoder_output)
        ...     print(result)
        "If you had scheduled some personal time..."
    """

    def __init__(
        self,
        version: str = "v3",
        quantization: QuantizationType = "int8",
        device: Literal["auto", "npu", "cuda", "cpu"] = "auto",
        backend: Literal["onnx", "pytorch"] = "pytorch",
        max_batch_size: int = 16,
        cache_dir: Optional[Path] = None,
    ):
        self.version = version
        self.quantization = quantization
        self.device = device
        self._requested_backend = backend  # Renamed to avoid conflict with property
        self.max_batch_size = max_batch_size
        self.cache_dir = cache_dir

        # Sessions (loaded on __enter__)
        self._prefix_session: Optional[Any] = None
        self._decoder_session: Optional[Any] = None
        self._pytorch_decoder: Optional[Any] = None  # PyTorch decoder head
        self._tokenizer: Optional[Any] = None
        self._loaded = False
        self._weights_path: Optional[Path] = None
        self._backend_name: str = "unknown"
        self._load_time_ms: float = 0.0

        # Generation config defaults
        self._default_max_tokens = 128
        self._default_temperature = 0.8
        self._default_top_p = 0.9
        self._default_top_k = 50
        self._default_repetition_penalty = 1.2

        # Special token IDs (ModernBERT tokenizer)
        self._pad_token_id = 50283
        self._bos_token_id = 50281  # CLS
        self._eos_token_id = 50282  # SEP

    def __enter__(self) -> "DecoderSession":
        """Load decoder into memory."""
        load_start = time.perf_counter()
        logger.info(f"Loading decoder v{self.version} (backend={self._requested_backend})...")

        try:
            if self._requested_backend == "pytorch":
                self._load_pytorch_decoder()
            else:
                self._load_onnx_decoder()

            # Try to load tokenizer
            self._load_tokenizer()

            self._loaded = True
            self._load_time_ms = (time.perf_counter() - load_start) * 1000

            memory_mb = self._estimate_memory_mb()
            logger.info(
                f"Decoder loaded in {self._load_time_ms:.0f}ms "
                f"(~{memory_mb:.0f} MB, backend: {self._backend_name})"
            )

        except Exception as e:
            logger.error(f"Failed to load decoder: {e}")
            # Clean up any partially loaded resources
            self._cleanup()
            raise RuntimeError(f"Failed to load decoder: {e}")

        return self

    def _load_pytorch_decoder(self) -> None:
        """Load PyTorch decoder from full model checkpoint."""
        from .weights_manager import download_decoder
        import torch

        # For PyTorch, we need fp32 weights (full model checkpoint)
        self._weights_path = download_decoder(
            version=self.version,
            quantization="fp32",  # Always fp32 for PyTorch
            cache_dir=self.cache_dir,
        )
        logger.info(f"Weights path: {self._weights_path}")

        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = self.device if self.device != "npu" else "cpu"

        self._backend_name = f"pytorch-{device}"
        logger.info(f"Using backend: {self._backend_name}")

        # Load the full model and extract decoder head
        try:
            from safetensors.torch import load_file

            # Load state dict
            model_path = self._weights_path / "model.safetensors"
            if not model_path.exists():
                raise FileNotFoundError(f"Model weights not found at {model_path}")

            state_dict = load_file(str(model_path))

            # Extract decoder weights (heads.counterfactual.*)
            decoder_weights = {}
            for key, value in state_dict.items():
                if key.startswith("heads.counterfactual."):
                    # Remove prefix
                    new_key = key.replace("heads.counterfactual.", "")
                    decoder_weights[new_key] = value

            logger.info(f"Extracted {len(decoder_weights)} decoder weight tensors")

            # Create decoder head with config
            from .models.decoder_gpt2 import GPT2DecoderHead
            from .models.decoder_gpt2_config import GPT2DecoderConfig

            config = GPT2DecoderConfig(
                gpt2_model_name="gpt2-medium",
                encoder_hidden_size=768,  # ModernBERT hidden size
            )
            self._pytorch_decoder = GPT2DecoderHead(
                config=config,
                encoder_hidden_size=768,
            )

            # Load weights
            missing, unexpected = self._pytorch_decoder.load_state_dict(
                decoder_weights, strict=False
            )
            if missing:
                logger.warning(f"Missing keys: {missing[:5]}...")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected[:5]}...")

            # Move to device
            self._pytorch_decoder = self._pytorch_decoder.to(device)
            self._pytorch_decoder.eval()

            logger.info(f"Loaded PyTorch decoder on {device}")

        except ImportError as e:
            raise RuntimeError(
                f"PyTorch decoder requires additional dependencies: {e}"
            )

    def _load_onnx_decoder(self) -> None:
        """Load ONNX decoder sessions."""
        from .weights_manager import download_decoder

        self._weights_path = download_decoder(
            version=self.version,
            quantization=self.quantization,
            cache_dir=self.cache_dir,
        )
        logger.info(f"Weights path: {self._weights_path}")

        # Get best provider based on device preference
        from .runtime import get_best_backend, Backend, ONNXSession

        if self.device == "auto":
            backend = get_best_backend()
        elif self.device == "npu":
            backend = Backend.DIRECTML
        elif self.device == "cuda":
            backend = Backend.CUDA
        else:
            backend = Backend.CPU

        self._backend_name = backend.value
        logger.info(f"Using backend: {self._backend_name}")

        # Load ONNX sessions
        prefix_path = self._weights_path / "prefix_encoder.onnx"
        decoder_path = self._weights_path / f"decoder_core_{self.quantization}.onnx"

        # Fallback to generic decoder path if quantization-specific doesn't exist
        if not decoder_path.exists():
            decoder_path = self._weights_path / "decoder_core.onnx"

        if prefix_path.exists():
            self._prefix_session = ONNXSession(prefix_path, backend=backend)
            logger.info(f"Loaded prefix encoder from {prefix_path}")
        else:
            logger.warning(f"Prefix encoder not found at {prefix_path}")

        if decoder_path.exists():
            self._decoder_session = ONNXSession(decoder_path, backend=backend)
            logger.info(f"Loaded decoder core from {decoder_path}")
        else:
            logger.warning(f"Decoder core not found at {decoder_path}")

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Unload decoder and free memory."""
        logger.info("Unloading decoder...")
        self._cleanup()
        logger.info("Decoder unloaded, memory freed")
        return False  # Don't suppress exceptions

    def _cleanup(self) -> None:
        """Clean up resources and free memory."""
        # Delete ONNX sessions
        if self._prefix_session is not None:
            del self._prefix_session
            self._prefix_session = None

        if self._decoder_session is not None:
            del self._decoder_session
            self._decoder_session = None

        # Delete PyTorch decoder
        if self._pytorch_decoder is not None:
            del self._pytorch_decoder
            self._pytorch_decoder = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._loaded = False

        # Force garbage collection
        gc.collect()

        # Also clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_tokenizer(self) -> None:
        """Load tokenizer for text encoding/decoding."""
        try:
            from transformers import AutoTokenizer

            # Try to load from weights path first
            tokenizer_path = self._weights_path / "tokenizer"
            if tokenizer_path.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                # Fallback to default ModernBERT tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "answerdotai/ModernBERT-base"
                )
            logger.debug("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Text decoding may fail.")
            self._tokenizer = None

    def _estimate_memory_mb(self) -> float:
        """Estimate current memory usage in MB."""
        # Rough estimates based on model size and quantization
        base_memory = {
            "fp32": 1400,
            "fp16": 700,
            "int8": 350,
        }
        return base_memory.get(self.quantization, 350)

    def _ensure_loaded(self) -> None:
        """Ensure decoder is loaded."""
        if not self._loaded:
            raise RuntimeError(
                "DecoderSession not loaded. Use 'with DecoderSession() as decoder:' "
                "context manager pattern."
            )

    @property
    def is_loaded(self) -> bool:
        """Check if decoder is currently loaded."""
        return self._loaded

    @property
    def backend(self) -> str:
        """Get the active backend name."""
        return self._backend_name

    @property
    def load_time_ms(self) -> float:
        """Get the time taken to load the decoder in milliseconds."""
        return self._load_time_ms

    def generate(
        self,
        encoder_hidden_states: np.ndarray,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate counterfactual text from encoder hidden states.

        This method takes the encoder's hidden state output and generates
        an alternative phrasing or counterfactual suggestion.

        Args:
            encoder_hidden_states: Encoder output array.
                Shape: (batch=1, seq_len, hidden_size=768)
            max_new_tokens: Maximum tokens to generate (default: 128)
            temperature: Sampling temperature (default: 0.8)
            top_p: Nucleus sampling probability (default: 0.9)
            top_k: Top-k sampling (default: 50)
            repetition_penalty: Repetition penalty (default: 1.2)

        Returns:
            Generated counterfactual text as a string.

        Raises:
            RuntimeError: If decoder is not loaded.

        Example:
            >>> encoder_output = client.encode("I hate this situation")
            >>> with DecoderSession() as decoder:
            ...     suggestion = decoder.generate(encoder_output)
            >>> print(suggestion)
            "I'm finding this situation challenging"
        """
        self._ensure_loaded()

        # Apply defaults
        max_new_tokens = max_new_tokens or self._default_max_tokens
        temperature = temperature if temperature is not None else self._default_temperature
        top_p = top_p if top_p is not None else self._default_top_p
        top_k = top_k if top_k is not None else self._default_top_k
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self._default_repetition_penalty

        # Ensure correct shape
        if encoder_hidden_states.ndim == 2:
            encoder_hidden_states = np.expand_dims(encoder_hidden_states, axis=0)

        # Generate tokens
        generated_ids = self._generate_tokens(
            encoder_hidden_states=encoder_hidden_states,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # Decode to text
        return self._decode_tokens(generated_ids)

    def generate_batch(
        self,
        encoder_hidden_states_list: List[np.ndarray],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        """
        Generate counterfactuals for multiple encoder outputs.

        For efficiency, consider batching inputs when processing multiple texts.

        Args:
            encoder_hidden_states_list: List of encoder outputs
            max_new_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling probability

        Returns:
            List of generated counterfactual texts
        """
        self._ensure_loaded()

        results = []
        for hidden_states in encoder_hidden_states_list:
            text = self.generate(
                hidden_states,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results.append(text)

        return results

    def generate_structured(
        self,
        encoder_hidden_states: np.ndarray,
        max_new_tokens: Optional[int] = None,
        extract_insights: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual with structured output.

        Returns a dictionary with the generated text and optional
        extracted procedural insights for P03 memory consolidation.

        Args:
            encoder_hidden_states: Encoder output array
            max_new_tokens: Maximum tokens to generate
            extract_insights: Whether to extract procedural insights

        Returns:
            Dictionary containing:
                - "text": The cleaned generated text
                - "raw": The raw generated text
                - "generation_time_ms": Time taken to generate
                - "procedural_insight": Optional extracted insight dict

        Example:
            >>> result = decoder.generate_structured(encoder_output)
            >>> print(result)
            {
                "text": "If you had scheduled 15 minutes...",
                "raw": "If you had scheduled 15 minutes of daily...",
                "generation_time_ms": 45.2,
                "procedural_insight": {
                    "trigger": "feeling overwhelmed",
                    "action": "schedule personal time",
                    "expected_outcome": "reduced stress"
                }
            }
        """
        self._ensure_loaded()

        start_time = time.perf_counter()
        text = self.generate(encoder_hidden_states, max_new_tokens=max_new_tokens)
        generation_time = (time.perf_counter() - start_time) * 1000

        result = {
            "text": self._clean_text(text),
            "raw": text,
            "generation_time_ms": round(generation_time, 2),
        }

        if extract_insights:
            result["procedural_insight"] = self._extract_insight(text)

        return result

    def _generate_tokens(
        self,
        encoder_hidden_states: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> np.ndarray:
        """
        Generate token IDs autoregressively.

        Routes to PyTorch or ONNX implementation based on backend.
        """
        if self._pytorch_decoder is not None:
            return self._generate_tokens_pytorch(
                encoder_hidden_states=encoder_hidden_states,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        else:
            return self._generate_tokens_onnx(
                encoder_hidden_states=encoder_hidden_states,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

    def _generate_tokens_pytorch(
        self,
        encoder_hidden_states: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> np.ndarray:
        """Generate tokens using PyTorch decoder."""
        import torch

        # Convert to torch tensor
        device = next(self._pytorch_decoder.parameters()).device
        hidden_states = torch.from_numpy(encoder_hidden_states).to(device)

        # Generate
        with torch.no_grad():
            output_ids = self._pytorch_decoder.generate(
                encoder_hidden_states=hidden_states,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
            )

        return output_ids.cpu().numpy()

    def _generate_tokens_onnx(
        self,
        encoder_hidden_states: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> np.ndarray:
        """
        Generate token IDs autoregressively using ONNX.

        This is the core generation loop that:
        1. Projects encoder outputs through prefix encoder
        2. Iteratively generates tokens using decoder core
        3. Applies sampling strategies (temperature, top-p, top-k)
        """
        # Check if ONNX sessions are available
        if self._prefix_session is None or self._decoder_session is None:
            logger.warning(
                "ONNX sessions not loaded. Returning placeholder generation. "
                "Ensure decoder ONNX files are exported and available."
            )
            # Return placeholder token sequence
            return np.array([[self._bos_token_id, self._eos_token_id]])

        try:
            # Step 1: Project encoder outputs to decoder space
            prefix_embeds = self._prefix_session.run(
                None,
                {"encoder_hidden_states": encoder_hidden_states.astype(np.float32)},
            )[0]

            batch_size = prefix_embeds.shape[0]
            prefix_len = prefix_embeds.shape[1]

            # Step 2: Initialize generation with BOS token
            generated_ids = np.full((batch_size, 1), self._bos_token_id, dtype=np.int64)
            finished = np.zeros(batch_size, dtype=bool)

            # Step 3: Autoregressive generation loop
            # Note: This ONNX model doesn't use KV cache, it takes full sequence each step
            for step in range(max_new_tokens):
                current_dec_len = generated_ids.shape[1]
                total_len = prefix_len + current_dec_len

                # Create attention mask (all ones - attending to everything)
                attention_mask = np.ones((batch_size, total_len), dtype=np.float32)

                # Run decoder with correct input names
                decoder_inputs = {
                    "prefix_embeds": prefix_embeds.astype(np.float32),
                    "decoder_input_ids": generated_ids,
                    "attention_mask": attention_mask,
                }

                outputs = self._decoder_session.run(None, decoder_inputs)
                logits = outputs[0][:, -1, :]  # (batch, vocab_size) - last token logits

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, generated_ids, repetition_penalty
                    )

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    logits = self._top_k_filter(logits, top_k)

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    logits = self._top_p_filter(logits, top_p)

                # Sample next token
                probs = self._softmax(logits)
                next_token = self._sample(probs)

                # Update finished mask
                finished = finished | (next_token.squeeze(-1) == self._eos_token_id)

                # Append to generated
                generated_ids = np.concatenate([generated_ids, next_token], axis=-1)

                # Stop if all sequences finished
                if finished.all():
                    break

            return generated_ids

        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            # Return minimal valid sequence on error
            return np.array([[self._bos_token_id, self._eos_token_id]])

    def _apply_repetition_penalty(
        self,
        logits: np.ndarray,
        generated_ids: np.ndarray,
        penalty: float,
    ) -> np.ndarray:
        """Apply repetition penalty to discourage repeating tokens."""
        for batch_idx in range(logits.shape[0]):
            unique_tokens = np.unique(generated_ids[batch_idx])
            for token_id in unique_tokens:
                if token_id < logits.shape[1]:
                    if logits[batch_idx, token_id] > 0:
                        logits[batch_idx, token_id] /= penalty
                    else:
                        logits[batch_idx, token_id] *= penalty
        return logits

    def _top_k_filter(self, logits: np.ndarray, k: int) -> np.ndarray:
        """Keep only top-k logits, set others to -inf."""
        if k <= 0:
            return logits

        # Get indices of top-k values
        top_k_indices = np.argsort(logits, axis=-1)[:, -k:]

        # Create mask
        mask = np.ones_like(logits, dtype=bool)
        for batch_idx in range(logits.shape[0]):
            mask[batch_idx, top_k_indices[batch_idx]] = False

        logits[mask] = float("-inf")
        return logits

    def _top_p_filter(self, logits: np.ndarray, p: float) -> np.ndarray:
        """Keep only tokens with cumulative probability <= p."""
        sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)

        cumulative_probs = np.cumsum(self._softmax(sorted_logits), axis=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        # Keep at least one token
        sorted_indices_to_remove[:, 0] = False

        # Scatter back
        for batch_idx in range(logits.shape[0]):
            remove_indices = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, remove_indices] = float("-inf")

        return logits

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _sample(self, probs: np.ndarray) -> np.ndarray:
        """Sample tokens from probability distribution."""
        batch_size = probs.shape[0]
        next_tokens = np.zeros((batch_size, 1), dtype=np.int64)

        for batch_idx in range(batch_size):
            next_tokens[batch_idx, 0] = np.random.choice(
                len(probs[batch_idx]),
                p=probs[batch_idx],
            )

        return next_tokens

    def _decode_tokens(self, token_ids: np.ndarray) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is None:
            logger.warning("Tokenizer not loaded, returning raw token IDs")
            return f"[Token IDs: {token_ids.tolist()}]"

        # Remove batch dimension if present
        if token_ids.ndim > 1:
            token_ids = token_ids[0]

        # Decode
        text = self._tokenizer.decode(
            token_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return text.strip()

    def _clean_text(self, text: str) -> str:
        """Clean generated text by removing artifacts."""
        # Remove common generation artifacts
        text = text.strip()

        # Remove repetitive patterns at the end
        lines = text.split("\n")
        if len(lines) > 1:
            # Keep only unique lines
            seen = set()
            unique_lines = []
            for line in lines:
                if line.strip() and line.strip() not in seen:
                    seen.add(line.strip())
                    unique_lines.append(line)
            text = "\n".join(unique_lines)

        return text

    def _extract_insight(self, text: str) -> Dict[str, str]:
        """
        Extract procedural insight from generated text.

        This is a simple pattern-based extraction. For production use,
        consider using a more sophisticated NLP approach or another model.
        """
        # Default insight structure
        insight = {
            "trigger": "",
            "action": "",
            "expected_outcome": "",
        }

        text_lower = text.lower()

        # Simple keyword-based extraction
        # Trigger patterns
        trigger_patterns = [
            "when feeling", "when you feel", "if feeling",
            "when stressed", "when overwhelmed", "when anxious",
        ]
        for pattern in trigger_patterns:
            if pattern in text_lower:
                # Extract text after the pattern
                idx = text_lower.index(pattern)
                end_idx = text.find(",", idx)
                if end_idx == -1:
                    end_idx = text.find(".", idx)
                if end_idx == -1:
                    end_idx = min(idx + 50, len(text))
                insight["trigger"] = text[idx:end_idx].strip()
                break

        # Action patterns
        action_patterns = [
            "try", "consider", "you could", "schedule",
            "take", "make time", "set aside",
        ]
        for pattern in action_patterns:
            if pattern in text_lower:
                idx = text_lower.index(pattern)
                end_idx = text.find(",", idx)
                if end_idx == -1:
                    end_idx = text.find(".", idx)
                if end_idx == -1:
                    end_idx = min(idx + 50, len(text))
                insight["action"] = text[idx:end_idx].strip()
                break

        # Outcome patterns
        outcome_patterns = [
            "to help", "to reduce", "to improve",
            "which helps", "resulting in", "leading to",
        ]
        for pattern in outcome_patterns:
            if pattern in text_lower:
                idx = text_lower.index(pattern)
                end_idx = text.find(".", idx)
                if end_idx == -1:
                    end_idx = min(idx + 50, len(text))
                insight["expected_outcome"] = text[idx:end_idx].strip()
                break

        return insight

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "version": self.version,
            "quantization": self.quantization,
            "backend": self._backend_name,
            "is_loaded": self._loaded,
            "load_time_ms": self._load_time_ms,
            "estimated_memory_mb": self._estimate_memory_mb() if self._loaded else 0,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_decoder_session(
    version: str = "v3",
    quantization: QuantizationType = "int8",
    device: Literal["auto", "npu", "cuda", "cpu"] = "auto",
) -> DecoderSession:
    """
    Create a new DecoderSession instance.

    This is a convenience function for creating decoder sessions
    without needing to import the class directly.

    Args:
        version: Decoder version
        quantization: Weight format
        device: Backend preference

    Returns:
        DecoderSession instance (use as context manager)

    Example:
        >>> from familyos_ultrabert.decoder_session import create_decoder_session
        >>> with create_decoder_session() as decoder:
        ...     result = decoder.generate(encoder_output)
    """
    return DecoderSession(
        version=version,
        quantization=quantization,
        device=device,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "DecoderSession",
    "create_decoder_session",
]
