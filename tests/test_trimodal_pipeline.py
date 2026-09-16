"""Unit tests for Trimodal Encoders, Dynamic Gated Cross-Attention, and MELD Feature Pipeline."""

import unittest
import torch

from src.foundation.config import Config
from src.ai_engine.unimodal.encoders import MockTextEncoder, MockAudioEncoder, MockVideoEncoder, FeatureEncoder
from src.ai_engine.unimodal.projections import LinearProjection
from src.ai_engine.multimodal.fusion import ConcatFusion, AverageFusion, DynamicGatedCrossAttentionFusion
from src.ai_engine.multimodal.classifiers import MLPClassifier
from src.ai_engine.builders.model import MERModel
from src.ai_engine.builders.builder import ModelBuilder
from src.ai_engine.dataset.components import SyntheticMELDDataset, MELDFeatureDataset, collate_multimodal_batch


class TestTrimodalPipeline(unittest.TestCase):
    """Test suite for trimodal architecture and proposed DGCA fusion."""

    def test_audio_encoder_and_feature_encoder(self):
        audio_enc = MockAudioEncoder(native_dim=768)
        self.assertEqual(audio_enc.modality, "audio")
        self.assertEqual(audio_enc.output_dim, 768)

        dummy_audio = torch.randn(4, 768)
        out = audio_enc(dummy_audio)
        self.assertEqual(out.shape, (4, 768))

        feat_enc = FeatureEncoder(native_dim=512, modality="video")
        self.assertEqual(feat_enc.output_dim, 512)
        out_feat = feat_enc(torch.randn(4, 512))
        self.assertEqual(out_feat.shape, (4, 512))

    def test_average_fusion(self):
        fusion = AverageFusion(projection_dim=256)
        features = {
            "text": torch.randn(4, 256),
            "audio": torch.randn(4, 256),
            "video": torch.randn(4, 256),
        }
        out = fusion(features)
        self.assertEqual(out.shape, (4, 256))

    def test_dynamic_gated_cross_attention_fusion(self):
        fusion = DynamicGatedCrossAttentionFusion(
            projection_dim=256,
            num_heads=4,
            dim_feedforward=512,
            modality_dropout=0.2,
        )
        features = {
            "text": torch.randn(4, 256),
            "audio": torch.randn(4, 256),
            "video": torch.randn(4, 256),
        }
        
        # Test eval mode (no modality dropout)
        fusion.eval()
        out = fusion(features)
        self.assertEqual(out.shape, (4, 256))
        self.assertIsNotNone(fusion.last_gating_weights)
        self.assertEqual(fusion.last_gating_weights.shape, (4, 3, 1))

        # Check weights sum to 1 across modalities (softmax verification for H3)
        weight_sum = fusion.last_gating_weights.sum(dim=1)
        self.assertTrue(torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5))

        # Test train mode (with modality dropout)
        fusion.train()
        out_train = fusion(features)
        self.assertEqual(out_train.shape, (4, 256))

    def test_full_trimodal_model_assembly(self):
        encoders = {
            "text": MockTextEncoder(native_dim=768),
            "audio": MockAudioEncoder(native_dim=768),
            "video": MockVideoEncoder(native_dim=512),
        }
        projections = {
            "text": LinearProjection(768, 256),
            "audio": LinearProjection(768, 256),
            "video": LinearProjection(512, 256),
        }
        fusion = DynamicGatedCrossAttentionFusion(projection_dim=256)
        classifier = MLPClassifier(input_dim=256, num_classes=7)

        model = MERModel(
            encoders=encoders,
            projections=projections,
            fusion=fusion,
            classifier=classifier,
        )

        batch_inputs = {
            "text": ["This is utterance 1", "This is utterance 2"],
            "audio": torch.randn(2, 768),
            "video": torch.randn(2, 512),
        }

        logits = model(batch_inputs)
        self.assertEqual(logits.shape, (2, 7))

    def test_model_builder_from_trimodal_yaml(self):
        config = Config.from_yaml("configs/meld_trimodal.yaml")
        model = ModelBuilder.build_model(config)
        self.assertIsInstance(model, MERModel)
        self.assertIn("text", model.encoders)
        self.assertIn("audio", model.encoders)
        self.assertIn("video", model.encoders)

    def test_meld_feature_dataset(self):
        ds = MELDFeatureDataset(data_dir="data/meld", split="train", num_samples=20)
        self.assertEqual(len(ds), 20)
        self.assertEqual(ds.num_classes, 7)
        sample = ds[0]
        self.assertIsNotNone(sample.text)
        self.assertIsNotNone(sample.audio)
        self.assertIsNotNone(sample.video)
        self.assertIsInstance(sample.label, int)

        batch = collate_multimodal_batch([ds[0], ds[1]])
        self.assertEqual(batch.inputs["text"].shape[0], 2)
        self.assertEqual(batch.inputs["audio"].shape[0], 2)
        self.assertEqual(batch.inputs["video"].shape[0], 2)
        self.assertEqual(batch.labels.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
