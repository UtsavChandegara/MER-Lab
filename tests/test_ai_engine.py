"""Unit & Contract tests for AI Engine components (MER-RULE-159)."""

import unittest
import torch

from src.ai_engine.dataset.components import SyntheticMELDDataset, collate_multimodal_batch
from src.ai_engine.unimodal.encoders import MockTextEncoder, MockVideoEncoder
from src.ai_engine.unimodal.projections import LinearProjection
from src.ai_engine.multimodal.fusion import ConcatFusion, AttentionFusion
from src.ai_engine.multimodal.classifiers import MLPClassifier
from src.ai_engine.builders.model import MERModel
from src.foundation.exceptions import ContractError


class TestAIEngineContracts(unittest.TestCase):

    def test_synthetic_dataset_and_batch_collation(self):
        dataset = SyntheticMELDDataset(num_samples=10, seed=42)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.num_classes, 7)

        sample = dataset[0]
        self.assertIsNotNone(sample.text)
        self.assertIsNotNone(sample.video)

        batch = collate_multimodal_batch([dataset[0], dataset[1]])
        self.assertEqual(len(batch.sample_ids), 2)
        self.assertIn("text", batch.inputs)
        self.assertIn("video", batch.inputs)
        self.assertEqual(batch.inputs["video"].shape, (2, 512))

    def test_mermodel_forward_pass_and_dimensions(self):
        text_enc = MockTextEncoder(native_dim=768)
        video_enc = MockVideoEncoder(native_dim=512)

        text_proj = LinearProjection(input_dim=768, output_dim=256)
        video_proj = LinearProjection(input_dim=512, output_dim=256)

        fusion = ConcatFusion(projection_dim=256, num_modalities=2)
        classifier = MLPClassifier(input_dim=256, num_classes=7)

        model = MERModel(
            encoders={"text": text_enc, "video": video_enc},
            projections={"text": text_proj, "video": video_proj},
            fusion=fusion,
            classifier=classifier,
        )

        inputs = {
            "text": ["UTTERANCE ONE", "UTTERANCE TWO"],
            "video": torch.randn(2, 512),
        }
        logits = model(inputs)
        self.assertEqual(logits.shape, (2, 7))

    def test_mermodel_contract_mismatch_raises_error(self):
        text_enc = MockTextEncoder(native_dim=768)
        # Invalid input dimension for projection (expected 768, given 512)
        invalid_proj = LinearProjection(input_dim=512, output_dim=256)
        fusion = ConcatFusion(projection_dim=256, num_modalities=1)
        classifier = MLPClassifier(input_dim=256, num_classes=7)

        with self.assertRaises(ContractError):
            MERModel(
                encoders={"text": text_enc},
                projections={"text": invalid_proj},
                fusion=fusion,
                classifier=classifier,
            )


if __name__ == "__main__":
    unittest.main()
