"""PyTorch correctness reference for the decoder-attention block."""

from .decoder_attention import (
    DecoderAttentionConfig,
    DecoderAttentionDebug,
    DecoderAttentionResult,
    apply_rope_reference,
    decoder_attention_reference,
    gqa_attention_reference,
    linear_reference,
    reshape_heads_reference,
    rms_norm_reference,
)

__all__ = [
    "DecoderAttentionConfig",
    "DecoderAttentionDebug",
    "DecoderAttentionResult",
    "apply_rope_reference",
    "decoder_attention_reference",
    "gqa_attention_reference",
    "linear_reference",
    "reshape_heads_reference",
    "rms_norm_reference",
]
