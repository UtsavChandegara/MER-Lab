"""Model Builder Service (MER-RULE-083..085, MER-RULE-127).

Instantiates component implementations dynamically from configuration file definitions.
"""

from typing import Dict
from src.foundation.config import Config
from src.foundation.logging import get_logger
from src.ai_engine.unimodal.contracts import BaseEncoder, BaseProjection
from src.ai_engine.multimodal.contracts import BaseFusion, BaseClassifier
from src.ai_engine.unimodal.registry import encoder_registry, projection_registry
from src.ai_engine.multimodal.registry import fusion_registry, classifier_registry
from src.ai_engine.builders.model import MERModel

logger = get_logger("MERLab.AIEngine.ModelBuilder")


class ModelBuilder:
    """Builder service responsible for constructing complete MERModel instances from Config."""

    @staticmethod
    def build_model(config: Config) -> MERModel:
        """Constructs an assembled MERModel from configuration settings.
        
        Args:
            config: Framework Config instance.
            
        Returns:
            MERModel: Fully constructed and contract-validated model ready for training/eval.
        """
        fusion_dim = config.get("model.fusion_dim", 256)
        
        # 1. Build Encoders & Projections per modality
        encoders: Dict[str, BaseEncoder] = {}
        projections: Dict[str, BaseProjection] = {}

        encoder_cfg = config["model"]["encoder"]
        for modality, cfg in encoder_cfg.items():
            enc_name = cfg["name"]
            raw_dim = cfg.get("raw_dim", 768)
            proj_type = cfg.get("projection_type", "linear")

            # Instantiate encoder
            encoder = encoder_registry.build(enc_name, native_dim=raw_dim)
            encoders[modality] = encoder

            # Instantiate corresponding projection standardizing to fusion_dim
            proj = projection_registry.build(
                proj_type,
                input_dim=encoder.output_dim,
                output_dim=fusion_dim,
            )
            projections[modality] = proj
            
            logger.info(
                f"Configured Modality '{modality}': Encoder='{enc_name}' (NativeDim={encoder.output_dim}) "
                f"-> Projection='{proj_type}' (TargetDim={fusion_dim})"
            )

        # 2. Build Fusion Module
        fusion_cfg = config["model"]["fusion"]
        fusion_name = fusion_cfg["name"]
        fusion_kwargs = {k: v for k, v in fusion_cfg.items() if k not in ["name", "projection_dim"]}
        fusion_module: BaseFusion = fusion_registry.build(
            fusion_name,
            projection_dim=fusion_dim,
            **fusion_kwargs,
        )
        logger.info(f"Configured Fusion Module: '{fusion_name}' (InputDim={fusion_dim}, OutputDim={fusion_module.output_dim})")

        # 3. Build Classifier Module
        classifier_cfg = config["model"]["classifier"]
        classifier_name = classifier_cfg["name"]
        num_classes = classifier_cfg.get("num_classes", 7)
        classifier_kwargs = {k: v for k, v in classifier_cfg.items() if k not in ["name", "num_classes"]}
        if classifier_name in ["mlp", "mlp_classifier"] and "hidden_dim" not in classifier_kwargs:
            classifier_kwargs["hidden_dim"] = 128

        classifier_module: BaseClassifier = classifier_registry.build(
            classifier_name,
            input_dim=fusion_module.output_dim,
            num_classes=num_classes,
            **classifier_kwargs,
        )
        logger.info(f"Configured Classifier Module: '{classifier_name}' (NumClasses={num_classes})")

        # Assemble and return MERModel
        model = MERModel(
            encoders=encoders,
            projections=projections,
            fusion=fusion_module,
            classifier=classifier_module,
        )
        logger.info("Successfully built and validated MERModel architecture.")
        return model
