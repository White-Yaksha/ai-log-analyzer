"""LLM Engine module for AI incident investigation.

Provides a lazy-loading wrapper around HuggingFace transformer models
with optional 4-bit quantization support.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LLMEngine:
    """Manages loading and inference for a causal language model.

    The model is lazily loaded on the first call to :meth:`generate`,
    keeping memory free until inference is actually needed.  When
    ``quantize`` is ``True`` the model is loaded in 4-bit precision via
    ``bitsandbytes``; if that library is unavailable the engine falls
    back to full-precision loading with a logged warning.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        quantize: bool = True,
        device: str = "auto",
    ) -> None:
        """Store configuration without loading the model.

        Args:
            model_name: HuggingFace model identifier or local path.
            quantize: Whether to apply 4-bit quantization via bitsandbytes.
            device: Device placement strategy passed as ``device_map``.
        """
        self.model_name: str = model_name
        self.quantize: bool = quantize
        self.device: str = device

        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the model and tokenizer into memory.

        If *quantize* is enabled, the model is loaded with 4-bit
        quantization (``BitsAndBytesConfig``).  When ``bitsandbytes`` is
        not installed the engine logs a warning and loads the model
        without quantization.
        """
        logger.info("Loading model '%s' …", self.model_name)

        quantization_config = None
        if self.quantize:
            try:
                import bitsandbytes as _bnb  # noqa: F401

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("4-bit quantization enabled via bitsandbytes.")
            except ImportError:
                logger.warning(
                    "bitsandbytes is not available — "
                    "falling back to loading without quantization."
                )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=self.device,
        )

        logger.info("Model '%s' loaded successfully.", self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return whether the model and tokenizer are currently loaded."""
        return self._model is not None and self._tokenizer is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from *prompt* using the underlying language model.

        The model and tokenizer are loaded lazily on the first
        invocation.  Generation runs inside a :func:`torch.no_grad`
        context for memory efficiency.

        Args:
            prompt: Input text to condition generation on.
            max_tokens: Maximum number of *new* tokens to generate.
            temperature: Sampling temperature (lower → more deterministic).
            top_p: Nucleus-sampling probability mass.

        Returns:
            The generated text with the input prompt stripped.
        """
        if not self.is_loaded():
            self._load_model()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        generated_tokens = outputs[0][input_length:]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
