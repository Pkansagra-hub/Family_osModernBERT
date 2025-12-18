"""
Example: P03 Dreaming Pipeline Integration

Shows how to use DecoderSession for R5 counterfactual generation.

The P03 system uses a multi-phase memory consolidation pipeline (R0-R8).
Most phases use the encoder for understanding/classification, but R5
("Dream Exploration") requires the decoder for counterfactual generation.

This example demonstrates memory-efficient decoder loading:
- Encoder stays resident throughout (175 MB with INT8)
- Decoder loaded only for R5 phase (adds ~350 MB)
- Decoder unloaded after R5 (back to 175 MB)

Total peak memory: ~525 MB
Sustained memory: ~175 MB
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    """A memory episode from P03."""
    
    id: str
    text: str
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class DreamingPipeline:
    """
    Example P03 dreaming pipeline with memory-efficient decoder usage.
    
    This class demonstrates the recommended pattern for integrating
    UltraBERT's counterfactual generation into the P03 nightly
    consolidation process.
    """

    def __init__(
        self,
        capabilities: Optional[List[str]] = None,
        backend: str = "onnx",
        quantization: str = "int8",
        decoder_quantization: str = "int8",
    ):
        """
        Initialize the dreaming pipeline.
        
        Args:
            capabilities: Encoder capabilities to load (default: standard set)
            backend: Inference backend ("onnx", "pytorch")
            quantization: Encoder quantization ("fp32", "fp16", "int8")
            decoder_quantization: Decoder quantization for R5 phase
        """
        # Import here to avoid import errors if not installed
        from familyos_ultrabert import Client
        
        # Default capabilities for P03 dreaming
        if capabilities is None:
            capabilities = [
                "sentiment",
                "emotions", 
                "topics",
                "entities",
                "relationship_type",
            ]
        
        # Encoder always resident (175 MB with INT8)
        self.client = Client(
            capabilities=capabilities,
            backend=backend,
            quantization=quantization,
        )
        
        self._decoder_quantization = decoder_quantization
        self._processed_episodes: List[Dict[str, Any]] = []

    def run_consolidation(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run full nightly consolidation pipeline.
        
        This runs the complete R0-R8 pipeline:
        - R0-R4: Encoder-only analysis and clustering
        - R5: Dream exploration (decoder for counterfactuals)
        - R6-R8: Integration and weight update
        
        Args:
            events: List of daily events (dicts with "text" key)
            
        Returns:
            Consolidation results with counterfactuals and insights
        """
        # R0-R4: Encoder-only phases
        episodes = self._run_r0_to_r4(events)

        # R5: Dream exploration (decoder loaded temporarily)
        counterfactuals = self._run_r5_dreams(episodes)

        # R6-R8: Encoder-only again
        results = self._run_r6_to_r8(episodes, counterfactuals)
        
        return results

    def _run_r0_to_r4(self, events: List[Dict[str, Any]]) -> List[Episode]:
        """
        R0-R4: Initial encoding and clustering phases.
        
        Uses encoder only - no decoder needed.
        """
        episodes = []
        
        for i, event in enumerate(events):
            text = event.get("text", "")
            if not text:
                continue
                
            # Create episode with analysis
            episode = Episode(
                id=f"ep_{i:04d}",
                text=text,
                timestamp=event.get("timestamp"),
                metadata={},
            )
            
            # Run encoder analysis
            result = self.client.analyze(text)
            
            # Store analysis in metadata
            episode.metadata = {
                "sentiment": result.get("sentiment", {}),
                "emotions": result.get("emotions", {}),
                "topics": result.get("topics", {}),
                "entities": result.get("entities", []),
            }
            
            episodes.append(episode)
        
        print(f"R0-R4: Processed {len(episodes)} episodes (encoder only)")
        return episodes

    def _run_r5_dreams(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """
        R5: Load decoder, generate counterfactuals, unload.
        
        This phase temporarily loads the decoder (~350 MB for INT8)
        to generate counterfactual alternatives for each episode.
        The decoder is automatically unloaded after this context exits.
        """
        counterfactuals = []
        
        # Identify episodes that need counterfactual exploration
        # (e.g., negative sentiment or high emotion intensity)
        dream_candidates = self._select_dream_candidates(episodes)
        
        if not dream_candidates:
            print("R5: No dream candidates found, skipping decoder load")
            return counterfactuals
        
        print(f"R5: Loading decoder for {len(dream_candidates)} candidates...")

        # Decoder loaded here, unloaded after context exits
        with self.client.create_decoder_session(
            quantization=self._decoder_quantization,
            device="auto",  # NPU -> CUDA -> CPU fallback
        ) as decoder:

            for episode in dream_candidates:
                # Get encoder representation
                encoder_output = self.client.encode(episode.text)

                # Generate structured counterfactual with insights
                result = decoder.generate_structured(encoder_output)

                counterfactuals.append({
                    "episode_id": episode.id,
                    "original": episode.text,
                    "alternative": result.get("text", ""),
                    "procedural_insight": result.get("procedural_insight", {}),
                    "generation_time_ms": result.get("generation_time_ms", 0),
                })
                
                print(f"  Generated counterfactual for {episode.id}")

        # Decoder automatically unloaded here
        # Memory: 525 MB -> 175 MB
        print(f"R5: Generated {len(counterfactuals)} counterfactuals, decoder unloaded")

        return counterfactuals

    def _select_dream_candidates(
        self,
        episodes: List[Episode],
        min_emotion_intensity: float = 0.3,
    ) -> List[Episode]:
        """
        Select episodes for dream exploration.
        
        Criteria:
        - Negative sentiment
        - High emotion intensity
        - Relationship-related topics
        """
        candidates = []
        
        for episode in episodes:
            metadata = episode.metadata or {}
            
            # Check sentiment
            sentiment = metadata.get("sentiment", {})
            if sentiment.get("label") == "negative":
                candidates.append(episode)
                continue
            
            # Check emotion intensity
            emotions = metadata.get("emotions", {})
            if isinstance(emotions, dict):
                max_intensity = max(emotions.values(), default=0)
                if max_intensity >= min_emotion_intensity:
                    candidates.append(episode)
                    continue
        
        return candidates

    def _run_r6_to_r8(
        self,
        episodes: List[Episode],
        counterfactuals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        R6-R8: Integration and final consolidation.
        
        Uses encoder only - decoder already unloaded.
        """
        # Build index of counterfactuals by episode ID
        cf_by_episode = {cf["episode_id"]: cf for cf in counterfactuals}
        
        # Merge counterfactuals into episodes
        enriched_episodes = []
        for episode in episodes:
            entry = {
                "id": episode.id,
                "text": episode.text,
                "analysis": episode.metadata,
            }
            
            if episode.id in cf_by_episode:
                cf = cf_by_episode[episode.id]
                entry["counterfactual"] = cf["alternative"]
                entry["procedural_insight"] = cf["procedural_insight"]
            
            enriched_episodes.append(entry)
        
        print(f"R6-R8: Consolidated {len(enriched_episodes)} episodes")
        
        return {
            "episodes": enriched_episodes,
            "counterfactual_count": len(counterfactuals),
            "total_episodes": len(episodes),
        }


def main():
    """Example usage of the DreamingPipeline."""
    
    # Create pipeline
    pipeline = DreamingPipeline(
        capabilities=["sentiment", "emotions", "topics", "entities"],
        quantization="int8",
        decoder_quantization="int8",
    )

    # Simulate daily events
    events = [
        {
            "text": "Had dinner with Mom at Luigi's",
            "timestamp": 1704067200.0,
        },
        {
            "text": "Felt overwhelmed with work deadlines",
            "timestamp": 1704070800.0,
        },
        {
            "text": "Kids argued about screen time",
            "timestamp": 1704074400.0,
        },
        {
            "text": "Nice walk in the park with the dog",
            "timestamp": 1704078000.0,
        },
        {
            "text": "Frustrated that nothing went right today",
            "timestamp": 1704081600.0,
        },
    ]

    # Run consolidation
    print("=" * 50)
    print("P03 Nightly Consolidation")
    print("=" * 50)
    
    results = pipeline.run_consolidation(events)
    
    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Counterfactuals generated: {results['counterfactual_count']}")
    
    # Show counterfactuals
    print("\nCounterfactuals:")
    for episode in results["episodes"]:
        if "counterfactual" in episode:
            print(f"\n  Original: {episode['text']}")
            print(f"  Alternative: {episode['counterfactual']}")
            if episode.get("procedural_insight"):
                insight = episode["procedural_insight"]
                print(f"  Insight: {insight.get('action', 'N/A')}")


if __name__ == "__main__":
    main()
