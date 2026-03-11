"""Tests for the LLMEngine module."""

import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing LLMEngine
# ---------------------------------------------------------------------------

_mock_torch = MagicMock()
_mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
_mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
_mock_torch.float16 = "float16"
sys.modules.setdefault("torch", _mock_torch)

_mock_transformers = MagicMock()
sys.modules.setdefault("transformers", _mock_transformers)

from src.llm_engine import LLMEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_engine_with_mock_model():
    """Create an LLMEngine and wire up mocked model + tokenizer."""
    engine = LLMEngine(model_name="test-model", quantize=False)

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Tokenizer: return fake input_ids
    fake_input_ids = MagicMock()
    fake_input_ids.shape = [1, 5]  # batch=1, seq_len=5
    tokenizer_output = MagicMock()
    tokenizer_output.__getitem__ = MagicMock(side_effect=lambda k: fake_input_ids if k == "input_ids" else MagicMock())
    tokenizer_output.to = MagicMock(return_value=tokenizer_output)
    mock_tokenizer.return_value = tokenizer_output

    # Model: generate returns tensor with input + new tokens
    fake_output_tensor = MagicMock()
    fake_output_tensor.__getitem__ = MagicMock(return_value=list(range(8)))
    mock_model.generate.return_value = [fake_output_tensor]
    mock_model.device = "cpu"

    # Tokenizer decode
    mock_tokenizer.decode.return_value = "Root cause: Kafka timeout."

    engine._model = mock_model
    engine._tokenizer = mock_tokenizer
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLazyLoading:
    def test_model_not_loaded_initially(self):
        engine = LLMEngine(model_name="test-model", quantize=False)
        assert not engine.is_loaded()

    @patch("src.llm_engine.AutoTokenizer")
    @patch("src.llm_engine.AutoModelForCausalLM")
    def test_model_loaded_on_generate(self, MockModel, MockTokenizer):
        engine = LLMEngine(model_name="test-model", quantize=False)

        mock_tokenizer_inst = MagicMock()
        fake_input_ids = MagicMock()
        fake_input_ids.shape = [1, 3]
        tok_output = MagicMock()
        tok_output.__getitem__ = MagicMock(side_effect=lambda k: fake_input_ids)
        tok_output.to = MagicMock(return_value=tok_output)
        mock_tokenizer_inst.return_value = tok_output
        mock_tokenizer_inst.decode.return_value = "analysis"
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_inst

        mock_model_inst = MagicMock()
        fake_out = MagicMock()
        fake_out.__getitem__ = MagicMock(return_value=list(range(6)))
        mock_model_inst.generate.return_value = [fake_out]
        mock_model_inst.device = "cpu"
        MockModel.from_pretrained.return_value = mock_model_inst

        assert not engine.is_loaded()
        engine.generate("test prompt")
        assert engine.is_loaded()


class TestGenerate:
    def test_returns_decoded_output(self):
        engine = _build_engine_with_mock_model()
        result = engine.generate("Analyze this error")
        assert isinstance(result, str)
        assert "Root cause" in result


class TestIsLoaded:
    def test_before_and_after_loading(self):
        engine = LLMEngine(model_name="m", quantize=False)
        assert engine.is_loaded() is False
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        assert engine.is_loaded() is True


class TestBitsAndBytesFallback:
    @patch("src.llm_engine.AutoTokenizer")
    @patch("src.llm_engine.AutoModelForCausalLM")
    def test_fallback_when_bnb_missing(self, MockModel, MockTokenizer):
        """When bitsandbytes is not importable, model loads without quantization."""
        engine = LLMEngine(model_name="test-model", quantize=True)

        MockTokenizer.from_pretrained.return_value = MagicMock()
        MockModel.from_pretrained.return_value = MagicMock()

        with patch.dict("sys.modules", {"bitsandbytes": None}):
            with patch("builtins.__import__", side_effect=ImportError("no bnb")):
                engine._load_model()

        # Model should still be loaded — without quantization_config
        MockModel.from_pretrained.assert_called_once()
        call_kwargs = MockModel.from_pretrained.call_args[1]
        assert call_kwargs.get("quantization_config") is None
