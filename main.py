"""Official Framework CLI Entry Point for MER-Lab (MER-RULE-205, MER-RULE-208).

Usage:
    python main.py --config configs/default.yaml
    python main.py --verify
"""

import sys
import argparse
from pathlib import Path

# Add src/ to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.foundation.logging import setup_logger, get_logger
from src.foundation.config import Config
from src.foundation.exceptions import MERLabException
from src.research.experiments.runner import ExperimentRunner

logger = setup_logger("MERLab")


def verify_infrastructure() -> bool:
    """Verifies infrastructure readiness (MER-RULE-208 Bootstrap Success Criteria)."""
    logger.info("Executing MER-Lab Infrastructure Verification (Phase 1 & 2)...")
    try:
        config_path = Path("configs/default.yaml")
        if not config_path.exists():
            logger.error(f"Verification failed: Default configuration file '{config_path}' not found.")
            return False

        config = Config.from_yaml(config_path)
        logger.info(f"Loaded Configuration: project.name='{config.get('project.name')}'")
        logger.info("Infrastructure verification PASSED! Ready for experiment execution.")
        return True
    except Exception as e:
        logger.error(f"Infrastructure verification failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="MER-Lab: Multimodal Emotion Recognition Research Framework CLI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["meld", "iemocap", "mosei"],
        default=None,
        help="Quick switch to run MELD (7-class), IEMOCAP (4-class), or CMU-MOSEI (6-class).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify framework infrastructure readiness and exit (MER-RULE-208).",
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_infrastructure()
        sys.exit(0 if success else 1)

    # Determine config file based on --dataset or --config
    config_path = args.config
    if args.dataset == "meld":
        config_path = "configs/meld_trimodal.yaml"
    elif args.dataset == "iemocap":
        config_path = "configs/iemocap_trimodal.yaml"
    elif args.dataset == "mosei":
        config_path = "configs/mosei_trimodal.yaml"
    elif config_path is None:
        config_path = "configs/default.yaml"

    logger.info(f"Starting MER-Lab experiment runner with config: '{config_path}' (Dataset: {args.dataset or 'config-specified'})")
    try:
        runner = ExperimentRunner(config_path)
        metrics = runner.run()
        print("\n================ FINAL EXPERIMENT RESULTS ================")
        print(f"Accuracy   : {metrics.get('accuracy', 0.0):.4f}")
        print(f"Weighted F1: {metrics.get('weighted_f1', 0.0):.4f}")
        print(f"Macro F1   : {metrics.get('macro_f1', 0.0):.4f}")
        print("==========================================================\n")
    except MERLabException as e:
        logger.error(f"MER-Lab Framework Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected Fatal Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
