"""Unit tests for Foundation Layer infrastructure (MER-RULE-158, MER-RULE-162)."""

import unittest
from pathlib import Path
from src.foundation.config import Config
from src.foundation.exceptions import MERLabException, ConfigurationError
from src.foundation.registry import Registry
from src.foundation.seed import set_seed
from src.foundation.device import get_device


class TestFoundation(unittest.TestCase):

    def test_config_loading_and_dot_notation(self):
        config_path = Path("configs/default.yaml")
        config = Config.from_yaml(config_path)
        self.assertIsNotNone(config.get("project.name"))
        self.assertEqual(config.get("model.fusion_dim"), 256)
        self.assertEqual(config.get("non_existent_key", default=999), 999)

    def test_config_missing_file_raises_error(self):
        with self.assertRaises(ConfigurationError):
            Config.from_yaml("configs/non_existent.yaml")

    def test_registry_registration_and_retrieval(self):
        reg = Registry("test_registry")
        
        @reg.register("dummy")
        class DummyClass:
            def __init__(self, val: int = 10):
                self.val = val

        self.assertIn("dummy", reg.list_modules())
        inst = reg.build("dummy", val=42)
        self.assertEqual(inst.val, 42)

    def test_seed_and_device(self):
        set_seed(123)
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
