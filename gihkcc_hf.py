"""
GIHKCC HuggingFace Integration

Hooks into any HuggingFace transformers model to intercept and compress
the KV cache during generation. Works with any model that uses the
standard DynamicCache or tuple-based KV cache format.

Usage:
    from gihkcc import GIHKCCConfig
    from gihkcc_hf import GIHKCCWrappedModel

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    config = GIHKCCConfig(l1_snr_threshold=0.92, l2_enabled=True)
    wrapped = GIHKCCWrappedModel(model, tokenizer, config)

    # Compress an existing cache
    result = wrapped.analyze_cache("The quick brown fox jumps over the lazy dog.")
    print(result["compression_summary"])

    # Generate with periodic cache compression
    output = wrapped.generate_compressed(
        "Once upon a time",
        max_new_tokens=200,
        compress_every=64,
    )
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Any, List, Tuple

import torch

logger = logging.getLogger("gihkcc.hf")


def _safe_import_transformers():
    """Import transformers with a clear error if missing."""
    try:
        import transformers
        return transformers
    except ImportError:
        raise ImportError(
            "HuggingFace transformers is required: pip install transformers"
        )


from gihkcc import (
    GIHKCCConfig,
    GIHKCCCompressedKVCache,
    compress_kv_cache,
    decompress_kv_cache,
    compute_snr_profile,
    estimate_memory_bytes,
)


class GIHKCCWrappedModel:
    """
    Wraps a HuggingFace causal LM to add GIHKCC KV cache compression.

    This does NOT modify the model weights or architecture. It intercepts
    the KV cache after a forward pass, compresses it, and can reconstruct
    it for subsequent generation steps.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[GIHKCCConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GIHKCCConfig()
        self.device = device or str(next(model.parameters()).device)

        # Cache state
        self._compressed_cache: Optional[GIHKCCCompressedKVCache] = None
        self._last_analysis: Optional[Dict[str, Any]] = None

    def extract_kv_cache(
        self,
        text: str,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Run a forward pass on text and extract the raw KV cache.

        Returns:
            (keys, values) where each is a list of tensors per layer.
            Each tensor shape: [num_heads, seq_len, head_dim]
        """
        transformers = _safe_import_transformers()

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                output_attentions=False,
            )

        past_kv = outputs.past_key_values

        keys = []
        values = []

        # Handle both DynamicCache and tuple formats
        if hasattr(past_kv, "key_cache"):
            # DynamicCache (transformers >= 4.36)
            for layer_idx in range(len(past_kv.key_cache)):
                # Shape: [batch, num_heads, seq_len, head_dim]
                k = past_kv.key_cache[layer_idx][0]  # Remove batch dim
                v = past_kv.value_cache[layer_idx][0]
                keys.append(k)
                values.append(v)
        else:
            # Tuple format: ((k, v), (k, v), ...)
            for layer_kv in past_kv:
                k, v = layer_kv
                keys.append(k[0])  # Remove batch dim
                values.append(v[0])

        return keys, values

    def analyze_cache(
        self,
        text: str,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract KV cache from text, analyze SNR profile, optionally compress.

        Returns a dict with SNR profiles, compression stats, and timing.
        """
        t0 = time.perf_counter()
        keys, values = self.extract_kv_cache(text)
        t_extract = time.perf_counter() - t0

        num_layers = len(keys)
        seq_len = keys[0].shape[-2] if keys else 0
        num_heads = keys[0].shape[0] if keys else 0
        head_dim = keys[0].shape[-1] if keys else 0

        # SNR profiles
        snr_k = compute_snr_profile(keys)
        snr_v = compute_snr_profile(values)

        result = {
            "num_layers": num_layers,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "snr_profile_keys": snr_k,
            "snr_profile_values": snr_v,
            "extraction_time_ms": round(t_extract * 1000, 2),
        }

        if compress:
            t1 = time.perf_counter()
            compressed = compress_kv_cache(keys, values, self.config)
            t_compress = time.perf_counter() - t1

            self._compressed_cache = compressed

            mem = estimate_memory_bytes(compressed)
            result["compression_summary"] = compressed.summary()
            result["memory"] = mem
            result["compression_time_ms"] = round(t_compress * 1000, 2)

            # Verify round-trip accuracy
            t2 = time.perf_counter()
            recon_keys, recon_values = decompress_kv_cache(compressed)
            t_decompress = time.perf_counter() - t2

            # Compute reconstruction error
            key_errors = []
            val_errors = []
            for i in range(num_layers):
                if recon_keys[i] is not None and keys[i] is not None:
                    err = (recon_keys[i] - keys[i]).abs().mean().item()
                    key_errors.append(err)
                if recon_values[i] is not None and values[i] is not None:
                    err = (recon_values[i] - values[i]).abs().mean().item()
                    val_errors.append(err)

            result["reconstruction"] = {
                "mean_key_error": round(sum(key_errors) / len(key_errors), 8) if key_errors else 0,
                "mean_value_error": round(sum(val_errors) / len(val_errors), 8) if val_errors else 0,
                "max_key_error": round(max(key_errors), 8) if key_errors else 0,
                "max_value_error": round(max(val_errors), 8) if val_errors else 0,
                "decompression_time_ms": round(t_decompress * 1000, 2),
            }

        self._last_analysis = result
        return result

    def generate_compressed(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        compress_every: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with periodic KV cache compression.

        Currently runs standard generation (the KV cache compression happens
        at analysis time). Full online compression during autoregressive
        generation requires deeper hooks into the model's forward loop —
        that's the next step.

        Args:
            prompt: Input text.
            max_new_tokens: Max tokens to generate.
            compress_every: Compress cache every N new tokens (planned).

        Returns:
            Dict with generated text, cache analysis, and timing.
        """
        transformers = _safe_import_transformers()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                **generate_kwargs,
            )
        t_gen = time.perf_counter() - t0

        generated_text = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Analyze the final cache state
        analysis = self.analyze_cache(full_text)

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "full_text": full_text,
            "generation_time_ms": round(t_gen * 1000, 2),
            "tokens_generated": output_ids.shape[1] - inputs.input_ids.shape[1],
            "cache_analysis": analysis,
        }

    def print_snr_heatmap(self, snr_profile: List[float], label: str = "SNR"):
        """Print a simple ASCII heatmap of SNR values."""
        bars = "░▒▓█"
        print(f"\n{label} Profile ({len(snr_profile)} transitions):")
        print("  Layer  SNR    Visual")
        print("  " + "-" * 40)
        for i, snr in enumerate(snr_profile):
            bar_idx = min(int(snr * len(bars)), len(bars) - 1)
            bar_len = int(snr * 30)
            bar = bars[bar_idx] * bar_len
            marker = " ◄ KF" if snr < self.config.l1_snr_threshold else ""
            print(f"  {i:3d}→{i+1:3d}  {snr:.4f}  {bar}{marker}")
