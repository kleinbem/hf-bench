"""Model architecture presets and exact GQA KV-cache sizing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantSpec:
    name: str
    gguf_type: str
    use_imatrix: bool
    publish_gguf: bool
    approx_bpw: float


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    hf_source: str
    hf_target_repo: str
    display_name: str
    params_b: float
    license_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_context: int

    def kv_bytes_per_token(self, kv_bits: float = 16.0) -> float:
        # 2 tensors (K and V) per layer * kv_heads * head_dim * bytes_per_elem.
        return 2.0 * self.num_layers * self.num_kv_heads * self.head_dim * (kv_bits / 8.0)

    def kv_mib(self, context_tokens: int, kv_bits: float = 16.0) -> float:
        return (self.kv_bytes_per_token(kv_bits) * context_tokens) / (1024.0 * 1024.0)


QUANT_MATRIX: tuple[QuantSpec, ...] = (
    QuantSpec("Q8_0", "Q8_0", use_imatrix=True, publish_gguf=True, approx_bpw=8.50),
    QuantSpec("Q6_K", "Q6_K", use_imatrix=True, publish_gguf=True, approx_bpw=6.56),
    QuantSpec("Q5_K_M", "Q5_K_M", use_imatrix=True, publish_gguf=True, approx_bpw=5.69),
    QuantSpec("Q4_K_M", "Q4_K_M", use_imatrix=True, publish_gguf=True, approx_bpw=4.85),
    # Uncalibrated control quantifies how much the agentic imatrix lowers KL-divergence.
    QuantSpec("Q4_K_M-std", "Q4_K_M", use_imatrix=False, publish_gguf=False, approx_bpw=4.85),
    QuantSpec("IQ4_XS", "IQ4_XS", use_imatrix=True, publish_gguf=True, approx_bpw=4.25),
    QuantSpec("IQ3_M", "IQ3_M", use_imatrix=True, publish_gguf=True, approx_bpw=3.44),
)

MODELS: dict[str, ModelSpec] = {
    "qwen3-4b": ModelSpec(
        slug="qwen3-4b",
        hf_source="Qwen/Qwen3-4B",
        hf_target_repo="kleinbem/Qwen3-4B-Edge-GGUF",
        display_name="Qwen3-4B",
        params_b=4.02,
        license_id="apache-2.0",
        num_layers=36,
        num_kv_heads=8,
        # Qwen3-4B explicitly sets head_dim=128 in config.json despite hidden_size=2560.
        head_dim=128,
        max_context=40960,
    ),
    "qwen2.5-coder-7b": ModelSpec(
        slug="qwen2.5-coder-7b",
        hf_source="Qwen/Qwen2.5-Coder-7B-Instruct",
        hf_target_repo="kleinbem/Qwen2.5-Coder-7B-Instruct-Edge-GGUF",
        display_name="Qwen2.5-Coder-7B-Instruct",
        params_b=7.62,
        license_id="apache-2.0",
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
        max_context=32768,
    ),
}
