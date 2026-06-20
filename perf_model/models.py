from __future__ import annotations

from dataclasses import dataclass


# Shared dataclasses keep the parser, roofline model, and simulators loosely coupled.
# Most fields are intentionally plain numbers so JSON/CSV exports stay easy to inspect.
@dataclass
class TirFacts:
    kernel_name: str
    arch: str | None
    grid_x: int
    grid_y: int
    threads_per_cta: int
    warps_per_cta: int
    num_ctas: int
    dynamic_shared_bytes_per_cta: int
    pipeline_stages: int | None
    c_elements: int | None
    textual_ptx_cp_async: int
    textual_ptx_ldmatrix: int
    textual_ptx_mma: int
    textual_ptx_commit_group: int
    textual_ptx_wait_group: int
    textual_tvm_storage_sync: int


@dataclass
class GemmModel:
    kernel_name: str
    arch: str | None
    M: int
    N: int
    K: int
    block_M: int
    block_N: int
    block_K: int
    k_tiles: int
    grid_x: int
    grid_y: int
    num_ctas: int
    threads_per_cta: int
    warps_per_cta: int
    dynamic_shared_bytes_per_cta: int
    pipeline_stages: int | None
    dtype_a: str
    dtype_b: str
    dtype_c: str
    bytes_a_per_element: int
    bytes_b_per_element: int
    bytes_c_per_element: int
    flops: int
    dram_read_bytes_tiled: int
    dram_write_bytes_tiled: int
    dram_bytes_tiled: int
    dram_bytes_ideal: int
    arithmetic_intensity_tiled: float
    arithmetic_intensity_ideal: float
    mma_shape: str | None
    mma_flops_per_instruction: int | None
    mma_instructions_total_est: int | None
    mma_instructions_per_cta_est: int | None
    roofline_bound_tflops: float | None
    memory_roof_tflops: float | None
    compute_roof_tflops: float | None
    notes: list[str]


@dataclass
class HardwareConfig:
    num_sms: int
    smem_per_sm_bytes: int
    max_threads_per_sm: int
    cta_limit_per_sm: int
    l2_capacity_bytes: int
    tensor_peak_tflops: float
    l2_bandwidth_gbs: float
    ddr_bandwidth_gbs: float
    smem_bandwidth_gbs: float
    tensor_efficiency: float
    l2_efficiency: float
    ddr_efficiency: float
    smem_efficiency: float


@dataclass
class PipelineLatencyModel:
    num_sms: int
    smem_per_sm_bytes: int
    max_threads_per_sm: int
    cta_limit_per_sm: int
    resident_ctas_per_sm: int
    active_ctas_per_wave: int
    waves: int
    tensor_peak_tflops: float
    load_bandwidth_gbs: float
    store_bandwidth_gbs: float
    tensor_efficiency: float
    load_efficiency: float
    store_efficiency: float
    effective_tensor_peak_tflops: float
    effective_load_bandwidth_gbs: float
    effective_store_bandwidth_gbs: float
    prologue_k_tiles: int
    steady_k_tiles: int
    epilogue_k_tiles: int
    load_bytes_per_k_tile_per_cta: int
    compute_flops_per_k_tile_per_cta: int
    store_bytes_per_cta: int
    load_us_per_k_tile: float
    compute_us_per_k_tile: float
    store_us_per_cta: float
    estimated_cta_us: float
    estimated_kernel_us: float
    estimated_kernel_ms: float
    estimated_tflops: float
    steady_state_bottleneck: str


@dataclass
class ResourceVectorUs:
    tc: float = 0.0
    smem: float = 0.0
    l2: float = 0.0
    ddr: float = 0.0

    def steady_time(self) -> float:
        # Independent resources are modeled as overlapped; the slowest lane sets time.
        return max(self.tc, self.smem, self.l2, self.ddr)

    def add(self, other: "ResourceVectorUs") -> "ResourceVectorUs":
        return ResourceVectorUs(
            tc=self.tc + other.tc,
            smem=self.smem + other.smem,
            l2=self.l2 + other.l2,
            ddr=self.ddr + other.ddr,
        )


@dataclass
class WaveResult:
    wave_index: int
    ctas: int
    active_sms: int
    resident_ctas_per_sm: int
    effective_depth: int
    l2_hit_rate: float
    load_l2_bytes: int
    load_ddr_bytes: int
    store_ddr_bytes: int
    load_vector_us_per_k: ResourceVectorUs
    compute_vector_us_per_k: ResourceVectorUs
    store_vector_us: ResourceVectorUs
    t_pro_us: float
    t_steady_us: float
    t_epi_us: float
    t_store_us: float
    latency_us: float
    steady_bottleneck: str


@dataclass
class SimulationResult:
    model_name: str
    kernel_name: str
    arch: str | None
    M: int
    N: int
    K: int
    block_M: int
    block_N: int
    block_K: int
    k_tiles: int
    grid_x: int
    grid_y: int
    num_ctas: int
    pipeline_stages: int
    resident_ctas_per_sm: int
    full_wave_capacity_ctas: int
    full_waves: int
    tail_ctas: int
    total_latency_us: float
    total_latency_ms: float
    estimated_tflops: float
    overall_l2_hit_rate: float
    hardware: HardwareConfig
    waves: list[WaveResult]
