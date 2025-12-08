#!/usr/bin/env python3
"""
Multi-Phase Training Orchestrator for ModernBERT v3

Orchestrates the complete training pipeline:
    Phase 0.5: Enhanced Healing (2,500 steps)
    Phase 1:   Multi-Task FamilyOS (10,000 steps)
    Phase 1.5: Forgetting Evaluation Gate
    Phase 2:   Fine-Tuning (optional, 5,000 steps)

The orchestrator handles:
    - Automatic phase transitions with model path chaining
    - State persistence for resume capability
    - Forgetting gate evaluation at Phase 1.5
    - Phase skipping for already-completed phases
    - W&B run naming per phase

Usage:
    # Run full pipeline from scratch
    python scripts/train_v3_orchestrator.py \\
        --start-phase 0.5 \\
        --end-phase 2 \\
        --output-dir outputs/v3_full

    # Run only Phase 0.5 and 1
    python scripts/train_v3_orchestrator.py \\
        --start-phase 0.5 \\
        --end-phase 1 \\
        --output-dir outputs/v3_training

    # Resume from Phase 1 with existing model
    python scripts/train_v3_orchestrator.py \\
        --resume-from outputs/v3_full/phase_0.5/best_model \\
        --start-phase 1 \\
        --end-phase 2

    # Force re-run of completed phases
    python scripts/train_v3_orchestrator.py \\
        --start-phase 0.5 \\
        --force-rerun

    # Dry run to see what would be executed
    python scripts/train_v3_orchestrator.py \\
        --start-phase 0.5 \\
        --dry-run

Output Structure:
    outputs/v3_full/
        orchestrator_state.json     # State for resume
        phase_0.5/
            best_model/             # Best checkpoint
            final_model/            # Final checkpoint
            results.json            # Training results
        phase_1/
            best_model/
            final_model/
            results.json
        phase_1.5/
            results.json            # Forgetting evaluation
            forgetting_report.json
            forgetting_report.md
        phase_2/
            best_model/
            final_model/
            results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Logging Setup
# =============================================================================

# Ensure unbuffered output for Colab/Jupyter compatibility
import os
os.environ["PYTHONUNBUFFERED"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override any existing config (needed for Colab)
)
logger = logging.getLogger(__name__)


# =============================================================================
# Phase Definitions
# =============================================================================


class Phase(Enum):
    """Training phases for v3."""

    PHASE_0_5 = "0.5"
    PHASE_1 = "1"
    PHASE_1_5 = "1.5"  # Forgetting evaluation gate
    PHASE_2 = "2"

    @classmethod
    def from_string(cls, s: str) -> Phase:
        """Create Phase from string."""
        mapping = {
            "0.5": cls.PHASE_0_5,
            "1": cls.PHASE_1,
            "1.5": cls.PHASE_1_5,
            "2": cls.PHASE_2,
        }
        if s not in mapping:
            raise ValueError(f"Unknown phase: {s}. Valid: {list(mapping.keys())}")
        return mapping[s]


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""

    name: str
    script: str
    config_file: str
    max_steps: int
    depends_on: str | None = None
    skip_if_exists: bool = True
    forgetting_gate: bool = False
    max_forgetting: float = 0.02  # 2% max allowed drop
    is_evaluation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "script": self.script,
            "config_file": self.config_file,
            "max_steps": self.max_steps,
            "depends_on": self.depends_on,
            "skip_if_exists": self.skip_if_exists,
            "forgetting_gate": self.forgetting_gate,
            "max_forgetting": self.max_forgetting,
            "is_evaluation": self.is_evaluation,
        }


# Default phase configurations
DEFAULT_PHASES: dict[str, PhaseConfig] = {
    "0.5": PhaseConfig(
        name="Enhanced Healing",
        script="scripts/v3_scripts/train_v3_phase0_5.py",
        config_file="configs/training/multitask/stage_v3_phase0_5_enhanced.yaml",
        max_steps=2500,
        depends_on=None,
    ),
    "1": PhaseConfig(
        name="Multi-Task FamilyOS",
        script="scripts/v3_scripts/train_v3_phase1.py",
        config_file="configs/training/multitask/stage_v3_phase1.yaml",
        max_steps=10000,
        depends_on="0.5",
    ),
    "1.5": PhaseConfig(
        name="Forgetting Evaluation",
        script="scripts/v3_scripts/evaluate_forgetting.py",
        config_file="configs/evaluation/forgetting_gate.yaml",
        max_steps=0,
        depends_on="1",
        forgetting_gate=True,
        max_forgetting=0.02,
        is_evaluation=True,
    ),
    "2": PhaseConfig(
        name="Fine-Tuning",
        script="scripts/v3_scripts/train_v3_phase2.py",  # Dedicated Phase 2 script
        config_file="configs/training/multitask/stage_v3_phase2.yaml",
        max_steps=5000,
        depends_on="1.5",
    ),
}


@dataclass
class OrchestratorConfig:
    """Full orchestrator configuration."""

    output_dir: str = "outputs/v3_full"
    start_phase: str = "0.5"
    end_phase: str = "2"
    resume_from: str | None = None

    # Phase configurations
    phases: dict[str, PhaseConfig] = field(default_factory=lambda: DEFAULT_PHASES.copy())

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "modernbert-v3"

    # Execution options
    force_rerun: bool = False
    dry_run: bool = False
    debug_run: bool = False  # Quick test: 5 steps per phase, limited samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "start_phase": self.start_phase,
            "end_phase": self.end_phase,
            "resume_from": self.resume_from,
            "phases": {k: v.to_dict() for k, v in self.phases.items()},
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "force_rerun": self.force_rerun,
            "dry_run": self.dry_run,
            "debug_run": self.debug_run,
        }


# =============================================================================
# Orchestrator State
# =============================================================================


@dataclass
class PhaseResult:
    """Result of a single phase execution."""

    phase_id: str
    status: str  # "completed", "failed", "skipped"
    start_time: str
    end_time: str
    elapsed_seconds: float
    output_dir: str
    model_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": self.elapsed_seconds,
            "output_dir": self.output_dir,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhaseResult:
        return cls(**d)


@dataclass
class OrchestratorState:
    """Persistent state for the orchestrator."""

    completed_phases: list[str] = field(default_factory=list)
    phase_results: dict[str, PhaseResult] = field(default_factory=dict)
    current_model_path: str | None = None
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_phases": self.completed_phases,
            "phase_results": {k: v.to_dict() for k, v in self.phase_results.items()},
            "current_model_path": self.current_model_path,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OrchestratorState:
        state = cls()
        state.completed_phases = d.get("completed_phases", [])
        state.phase_results = {
            k: PhaseResult.from_dict(v) for k, v in d.get("phase_results", {}).items()
        }
        state.current_model_path = d.get("current_model_path")
        state.last_updated = d.get("last_updated", "")
        return state

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        from datetime import datetime

        self.last_updated = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> OrchestratorState:
        """Load state from JSON file."""
        if not path.exists():
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Training Orchestrator
# =============================================================================


class TrainingOrchestrator:
    """
    Orchestrates multi-phase training with automatic transitions.

    The orchestrator manages:
    - Phase execution order
    - Model path chaining between phases
    - State persistence for resume
    - Forgetting gate evaluation
    - W&B integration
    """

    ALL_PHASES = ["0.5", "1", "1.5", "2"]

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State management
        self.state_file = self.output_dir / "orchestrator_state.json"
        self.state = OrchestratorState.load(self.state_file)

        # Set initial model path if resuming
        if config.resume_from:
            self.state.current_model_path = config.resume_from

    def _save_state(self) -> None:
        """Save orchestrator state to disk."""
        self.state.save(self.state_file)

    def _get_phases_to_run(self) -> list[str]:
        """Determine which phases to run based on config."""
        start_idx = self.ALL_PHASES.index(self.config.start_phase)
        end_idx = self.ALL_PHASES.index(self.config.end_phase)
        return self.ALL_PHASES[start_idx : end_idx + 1]

    def _should_skip_phase(self, phase_id: str) -> bool:
        """Check if phase should be skipped."""
        if self.config.force_rerun:
            return False

        phase_config = self.config.phases.get(phase_id)
        if not phase_config:
            return True

        # Check if already completed
        if phase_id in self.state.completed_phases:
            if phase_config.skip_if_exists:
                # Verify output exists
                phase_output = self.output_dir / f"phase_{phase_id}"
                if phase_config.is_evaluation:
                    return (phase_output / "results.json").exists()
                else:
                    return (phase_output / "final_model").exists() or (
                        phase_output / "best_model"
                    ).exists()
        return False

    def _get_model_path_for_phase(self, phase_id: str) -> str | None:
        """Get the model path to use for a phase."""
        phase_config = self.config.phases.get(phase_id)

        # Special case: Phase 2 depends on 1.5 for execution order,
        # but uses Phase 1's model (1.5 is evaluation only)
        if phase_id == "2" and phase_config and phase_config.depends_on == "1.5":
            # Force use of Phase 1's model, ignoring current_model_path
            prev_output = self.output_dir / "phase_1"
            for model_name in ["best_model", "final_model"]:
                model_path = prev_output / model_name
                if model_path.exists():
                    return str(model_path)

        # If we have a current model path, use it
        if self.state.current_model_path:
            return self.state.current_model_path

        # Otherwise, look for previous phase output
        if phase_config and phase_config.depends_on:
            prev_phase = phase_config.depends_on
            prev_output = self.output_dir / f"phase_{prev_phase}"

            # Check for model in previous phase
            for model_name in ["best_model", "final_model"]:
                model_path = prev_output / model_name
                if model_path.exists():
                    return str(model_path)

        return None

    def _build_command(
        self,
        phase_id: str,
        phase_config: PhaseConfig,
        model_path: str | None,
        phase_output: Path,
    ) -> list[str]:
        """Build the command to execute a phase."""
        cmd = [
            sys.executable,
            phase_config.script,
            "--config",
            phase_config.config_file,
            "--output-dir",
            str(phase_output),
        ]

        # Add model path if available
        if model_path:
            cmd.extend(["--model-path", model_path])

        # Handle debug run mode: override steps and enable debug flag
        if self.config.debug_run and not phase_config.is_evaluation:
            cmd.extend(["--max-steps", "5"])
            cmd.append("--debug")
        elif phase_config.max_steps > 0:
            cmd.extend(["--max-steps", str(phase_config.max_steps)])

        # Add wandb options (disable for debug runs)
        if self.config.debug_run:
            cmd.append("--no-wandb")
        elif self.config.use_wandb:
            cmd.extend(["--wandb-run-name", f"v3_phase_{phase_id}"])
        else:
            cmd.append("--no-wandb")

        # For forgetting evaluation, add baseline path
        if phase_config.forgetting_gate:
            baseline_path = self.output_dir / "phase_0.5" / "best_model"
            if baseline_path.exists():
                cmd.extend(["--baseline", str(baseline_path)])

        return cmd

    def _run_phase(
        self,
        phase_id: str,
        phase_config: PhaseConfig,
        model_path: str | None,
    ) -> PhaseResult:
        """Execute a single phase."""
        from datetime import datetime

        phase_output = self.output_dir / f"phase_{phase_id}"
        phase_output.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()
        start_time_str = start_time.isoformat()

        # Build command
        cmd = self._build_command(phase_id, phase_config, model_path, phase_output)

        logger.info(f"Executing: {' '.join(cmd)}")

        if self.config.dry_run:
            logger.info("[DRY RUN] Would execute above command")
            return PhaseResult(
                phase_id=phase_id,
                status="dry_run",
                start_time=start_time_str,
                end_time=start_time_str,
                elapsed_seconds=0,
                output_dir=str(phase_output),
            )

        # Execute command - stream output in real-time for visibility
        try:
            # Set up environment for unbuffered Python output (critical for Colab)
            import os
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                cwd=str(Path(__file__).parent.parent.parent),  # Run from project root
                env=env,  # Pass environment with unbuffered flag
            )

            # Stream output in real-time
            print(f"\n{'='*60}")
            print(f"[PHASE {phase_id}] Output:")
            print(f"{'='*60}")

            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(line, end="", flush=True)  # Print immediately

            process.wait()

            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()

            print(f"{'='*60}")
            print(f"[PHASE {phase_id}] Finished with code {process.returncode}")
            print(f"{'='*60}\n")

            if process.returncode != 0:
                logger.error(f"Phase {phase_id} failed with code {process.returncode}")

                return PhaseResult(
                    phase_id=phase_id,
                    status="failed",
                    start_time=start_time_str,
                    end_time=end_time.isoformat(),
                    elapsed_seconds=elapsed,
                    output_dir=str(phase_output),
                    error=f"Exit code {process.returncode}",
                )

            # Load results if available
            results_file = phase_output / "results.json"
            metrics = {}
            output_model_path = None

            if results_file.exists():
                with open(results_file) as f:
                    results_data = json.load(f)
                    metrics = results_data
                    # Only set output_model_path if this is NOT an evaluation phase
                    if not phase_config.is_evaluation:
                        output_model_path = results_data.get("output_dir")

            # Find model path in output (only for non-evaluation phases)
            if not output_model_path and not phase_config.is_evaluation:
                for model_name in ["best_model", "final_model"]:
                    candidate = phase_output / model_name
                    if candidate.exists():
                        output_model_path = str(candidate)
                        break

            logger.info(f"Phase {phase_id} completed in {elapsed/60:.1f} minutes")

            return PhaseResult(
                phase_id=phase_id,
                status="completed",
                start_time=start_time_str,
                end_time=end_time.isoformat(),
                elapsed_seconds=elapsed,
                output_dir=str(phase_output),
                model_path=output_model_path,
                metrics=metrics,
            )

        except Exception as e:
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()

            logger.exception(f"Phase {phase_id} failed with exception")

            return PhaseResult(
                phase_id=phase_id,
                status="failed",
                start_time=start_time_str,
                end_time=end_time.isoformat(),
                elapsed_seconds=elapsed,
                output_dir=str(phase_output),
                error=str(e),
            )

    def _check_forgetting_gate(self, result: PhaseResult, phase_config: PhaseConfig) -> bool:
        """Check if forgetting gate passed."""
        if "forgetting_metrics" not in result.metrics:
            logger.warning("No forgetting metrics found, skipping gate check")
            return True

        metrics = result.metrics["forgetting_metrics"]
        max_drop = phase_config.max_forgetting

        all_passed = True
        for task, drop in metrics.items():
            if drop > max_drop:
                logger.error(f"Forgetting exceeded for {task}: {drop:.2%} > {max_drop:.2%}")
                all_passed = False
            else:
                logger.info(f"Forgetting OK for {task}: {drop:.2%} <= {max_drop:.2%}")

        return all_passed

    def run(self) -> dict[str, Any]:
        """
        Run the full training pipeline.

        Returns:
            Dict with status and results from all phases.
        """
        logger.info("=" * 70)
        logger.info("ModernBERT v3 Multi-Phase Training Orchestrator")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Start phase: {self.config.start_phase}")
        logger.info(f"End phase: {self.config.end_phase}")

        if self.config.resume_from:
            logger.info(f"Resuming from: {self.config.resume_from}")

        if self.config.dry_run:
            logger.info("[DRY RUN MODE]")

        # Determine phases to run
        phases_to_run = self._get_phases_to_run()
        logger.info(f"Phases to run: {phases_to_run}")

        # Save initial config
        config_file = self.output_dir / "orchestrator_config.json"
        with open(config_file, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Run each phase
        for phase_id in phases_to_run:
            phase_config = self.config.phases.get(phase_id)
            if not phase_config:
                logger.error(f"No configuration for phase {phase_id}")
                continue

            # Check if should skip
            if self._should_skip_phase(phase_id):
                logger.info(f"Phase {phase_id} already completed, skipping")

                # Update model path from previous result
                if phase_id in self.state.phase_results:
                    prev_result = self.state.phase_results[phase_id]
                    if prev_result.model_path:
                        self.state.current_model_path = prev_result.model_path

                continue

            # Get model path for this phase
            model_path = self._get_model_path_for_phase(phase_id)

            # Log phase start
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Starting Phase {phase_id}: {phase_config.name}")
            logger.info("=" * 70)

            if model_path:
                logger.info(f"Using model: {model_path}")

            # Run phase
            result = self._run_phase(phase_id, phase_config, model_path)

            # Store result
            self.state.phase_results[phase_id] = result

            # Check result
            if result.status == "failed":
                logger.error(f"Phase {phase_id} FAILED")
                self._save_state()

                return {
                    "status": "failed",
                    "failed_phase": phase_id,
                    "error": result.error,
                    "completed_phases": self.state.completed_phases,
                    "results": {k: v.to_dict() for k, v in self.state.phase_results.items()},
                }

            # Check forgetting gate
            if phase_config.forgetting_gate:
                gate_passed = self._check_forgetting_gate(result, phase_config)
                if not gate_passed:
                    logger.error(f"Forgetting gate FAILED at Phase {phase_id}")
                    logger.error("Consider increasing replay ratio and re-running Phase 1")
                    self._save_state()

                    return {
                        "status": "failed",
                        "failed_phase": phase_id,
                        "reason": "forgetting_gate",
                        "completed_phases": self.state.completed_phases,
                        "results": {k: v.to_dict() for k, v in self.state.phase_results.items()},
                    }

            # Update state
            self.state.completed_phases.append(phase_id)
            if result.model_path:
                self.state.current_model_path = result.model_path

            self._save_state()

        # Success
        logger.info("")
        logger.info("=" * 70)
        logger.info("All phases completed successfully!")
        logger.info("=" * 70)

        # Print summary
        total_time = sum(
            r.elapsed_seconds for r in self.state.phase_results.values() if r.elapsed_seconds
        )
        logger.info(f"Total training time: {total_time/3600:.2f} hours")
        logger.info(f"Final model: {self.state.current_model_path}")

        return {
            "status": "success",
            "completed_phases": self.state.completed_phases,
            "final_model_path": self.state.current_model_path,
            "total_time_seconds": total_time,
            "results": {k: v.to_dict() for k, v in self.state.phase_results.items()},
        }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ModernBERT v3 Multi-Phase Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python scripts/train_v3_orchestrator.py --start-phase 0.5 --end-phase 2

    # Run Phase 0.5 and 1 only
    python scripts/train_v3_orchestrator.py --start-phase 0.5 --end-phase 1

    # Resume from existing model
    python scripts/train_v3_orchestrator.py --resume-from outputs/v3/phase_0.5/best_model --start-phase 1

    # Dry run
    python scripts/train_v3_orchestrator.py --dry-run
""",
    )

    # Phase selection
    parser.add_argument(
        "--start-phase",
        type=str,
        default="0.5",
        choices=["0.5", "1", "1.5", "2"],
        help="Phase to start from (default: 0.5)",
    )
    parser.add_argument(
        "--end-phase",
        type=str,
        default="2",
        choices=["0.5", "1", "1.5", "2"],
        help="Phase to end at (default: 2)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/v3_full",
        help="Base output directory",
    )

    # Resume
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Model path to resume from",
    )

    # Execution options
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-run of completed phases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--debug-run",
        action="store_true",
        help="Quick debug: 5 steps/phase with limited samples",
    )

    # W&B
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="modernbert-v3",
        help="W&B project name",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Build config
    config = OrchestratorConfig(
        output_dir=args.output_dir,
        start_phase=args.start_phase,
        end_phase=args.end_phase,
        resume_from=args.resume_from,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        force_rerun=args.force_rerun,
        dry_run=args.dry_run,
        debug_run=args.debug_run,
    )

    # Log debug mode
    if config.debug_run:
        logger.info("[DEBUG RUN] 5 steps per phase, limited samples, no W&B")

    # Create orchestrator
    orchestrator = TrainingOrchestrator(config)

    # Run pipeline
    results = orchestrator.run()

    # Return exit code
    if results["status"] == "success":
        print("\n[SUCCESS] Training pipeline completed successfully!")
        print(f"Final model: {results.get('final_model_path', 'N/A')}")
        return 0
    else:
        print(f"\n[FAILED] Training pipeline failed at phase {results.get('failed_phase')}")
        if results.get("reason") == "forgetting_gate":
            print("Reason: Forgetting gate threshold exceeded")
            print("Suggested action: Increase replay ratio and re-run Phase 1")
        return 1


if __name__ == "__main__":
    sys.exit(main())
