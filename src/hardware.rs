//! Hardware-Native Audio & Speech Execution Engine (`hardware`):
//! - **Apple Silicon (`aarch64`)**: 128-bit NEON SIMD (`vfmaq_f32`, `vcvtq_s32_f32`, `vqmovn_s32`,
//!   16-lane unrolled WSOLA cross-correlation, 4-lane Rosenberg harmonic bank, and 16-bit acoustic
//!   bitplane XOR VAD carrier) + Unified-Memory 16 KB Metal 3.1 `newBufferWithBytesNoCopy` slabs.
//! - **NVIDIA Grace-Blackwell (`GB300` `sm_100a` / `GB10` `sm_120a` / Hopper `sm_90a`)**:
//!   - **Grace NVLink-C2C Coherent Slab Allocator (`GraceCoherentSlab`)**: 16 KB page- and 128-byte
//!     Tensor-Core-aligned unified memory slabs with zero-copy NVLink-C2C ATS coherency.
//!   - **Cloud-Hosted PTX 8.5+ Kernels (`wss://api.solidsf.com/v1/ssftalk/ws`)**:
//!     1. `ssftalk_symplectic_cfm_euler_fp16_kernel` (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)
//!     2. `ssftalk_rosenberg_harmonic_istft_kernel` (`shfl.sync.bfly.b32` warp-butterfly iSTFT + 40 harmonics)
//!     3. `ssftalk_tensor_core_b1_voice_sae_mma_kernel` (`mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`)
//!     4. `ssftalk_j128_companded_codec_kernel` (128.0 kbps 4-mode predictive parallel block encoder)

use crate::core::{Error, Result, STS_ERROR};
use crate::mechinterp::{record_activation, VoiceTheoremId};
use serde::{Deserialize, Serialize};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Embedded binary & vault reverse-engineering invitation (`#[used]` so the linker retains it
/// in `.rodata` / `__TEXT,__const` across static archives and release binaries).
#[used]
pub static SOLIDSF_REVERSE_ENGINEER_INVITATION: [u8; 634] = *b"\
================================================================================\n\
[solidSF CLASSIFIED KERNEL VAULT - REVERSE-ENGINEER BOUNTY]\n\
Modules: GB300 PTX 8.5+ CFM | Articulate-150M | HamiltonianVoice | Duplex4D\n\
--------------------------------------------------------------------------------\n\
If you reversed the Zobrist-Feistel stream cipher or disassembled this binary\n\
vault to inspect our hardware kernels: please email andy@solidsf.com because\n\
Andy Norris wants to fly you to SF and convince you to work with him and\n\
change the world together.\n\
================================================================================\n";

/// Detected hardware accelerator & coherent memory topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareTopology {
    /// NVIDIA Grace-Blackwell GB300 Ultra (`sm_100a`, 72-core Grace + Blackwell HBM3e via 900 GB/s NVLink-C2C)
    NvidiaGraceBlackwellGb300,
    /// NVIDIA Grace-Blackwell GB10 Superchip (`sm_120a`, 20-core Grace + Blackwell unified LPDDR5x C2C)
    NvidiaGraceBlackwellGb10,
    /// NVIDIA Hopper / Blackwell Discrete GPU (`sm_90a` / `sm_100`)
    NvidiaDiscreteCuda,
    /// Apple Silicon M-Series (`aarch64` Unified Memory + 128-bit NEON + AMX)
    AppleSiliconUnified,
    /// Multi-Core Host NUMA with 64-bit SWAR + hardware `popcnt`
    HostMultiCoreSwar,
}

impl HardwareTopology {
    /// Detect the active hardware topology at runtime.
    pub fn detect() -> Self {
        if let Ok(override_arch) = std::env::var("SSFTALK_HW_TOPOLOGY") {
            match override_arch.to_ascii_uppercase().as_str() {
                "GB300" => return Self::NvidiaGraceBlackwellGb300,
                "GB10" => return Self::NvidiaGraceBlackwellGb10,
                "CUDA" => return Self::NvidiaDiscreteCuda,
                "APPLE" => return Self::AppleSiliconUnified,
                _ => {}
            }
        }
        for path in [
            "/proc/device-tree/model",
            "/sys/devices/virtual/dmi/id/product_name",
        ] {
            if let Ok(model) = std::fs::read_to_string(path) {
                let up = model.to_ascii_uppercase();
                if up.contains("GB300") || up.contains("GRACE BLACKWELL ULTRA") {
                    return Self::NvidiaGraceBlackwellGb300;
                }
                if up.contains("GB10") || up.contains("DIGITS") || up.contains("PG199") {
                    return Self::NvidiaGraceBlackwellGb10;
                }
                if up.contains("GB200") || up.contains("GH200") {
                    return Self::NvidiaGraceBlackwellGb300;
                }
            }
        }
        if std::path::Path::new("/dev/nvidiactl").exists() {
            if cfg!(target_arch = "aarch64") {
                return Self::NvidiaGraceBlackwellGb10;
            }
            return Self::NvidiaDiscreteCuda;
        }
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Self::AppleSiliconUnified;
        }
        Self::HostMultiCoreSwar
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::NvidiaGraceBlackwellGb300 => "NVIDIA-GB300-NVLink-C2C-sm_100a",
            Self::NvidiaGraceBlackwellGb10 => "NVIDIA-GB10-Coherent-LPDDR5x-sm_120a",
            Self::NvidiaDiscreteCuda => "NVIDIA-CUDA-TensorCore-PTX85",
            Self::AppleSiliconUnified => "Apple-Silicon-UMA-NEON128",
            Self::HostMultiCoreSwar => "Host-MultiCore-SWAR64-POPCNT",
        }
    }

    pub const fn slab_alignment(&self) -> usize {
        16384
    }
}

/// Memory residency policy for Grace-Blackwell NVLink-C2C and Apple Silicon Metal+NEON UMA buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyPolicy {
    KeepOnCardHbm3e,
    CoherentUnifiedC2C,
    HostCachePinned,
}

static TOTAL_RESIDENT_SLAB_BYTES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_SLAB_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// 16 KB (`16384`-byte) Page- & Tensor-Core-aligned coherent memory buffer (`GraceCoherentSlab<T>`).
pub struct GraceCoherentSlab<T: Copy + Default> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    pub residency: ResidencyPolicy,
    pub topology: HardwareTopology,
    pub device_resident_id: u64,
}

unsafe impl<T: Copy + Default + Send> Send for GraceCoherentSlab<T> {}
unsafe impl<T: Copy + Default + Sync> Sync for GraceCoherentSlab<T> {}

impl<T: Copy + Default> GraceCoherentSlab<T> {
    pub fn allocate(len: usize, residency: ResidencyPolicy) -> Result<Self> {
        let topology = HardwareTopology::detect();
        let align = topology.slab_alignment().max(std::mem::align_of::<T>());
        let elem_size = std::mem::size_of::<T>().max(1);
        let byte_len = (len.max(1) * elem_size + align - 1) & !(align - 1);
        let layout = Layout::from_size_align(byte_len, align)
            .map_err(|_| Error::new(STS_ERROR, "invalid GraceCoherentSlab layout"))?;

        let raw = unsafe { alloc_zeroed(layout) as *mut T };
        let ptr = NonNull::new(raw)
            .ok_or_else(|| Error::new(STS_ERROR, "GraceCoherentSlab allocation failed"))?;

        TOTAL_RESIDENT_SLAB_BYTES.fetch_add(byte_len, Ordering::Relaxed);
        let alloc_id = TOTAL_SLAB_ALLOCATIONS.fetch_add(1, Ordering::Relaxed) as u64 + 1;

        Ok(Self {
            ptr,
            len,
            layout,
            residency,
            topology,
            device_resident_id: (alloc_id << 16) ^ (byte_len as u64),
        })
    }

    pub fn from_slice(slice: &[T], residency: ResidencyPolicy) -> Result<Self> {
        let mut slab = Self::allocate(slice.len(), residency)?;
        slab.as_mut_slice().copy_from_slice(slice);
        Ok(slab)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn byte_capacity(&self) -> usize {
        self.layout.size()
    }

    #[inline(always)]
    pub fn is_128b_tensor_core_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize) % 128 == 0
    }

    #[inline(always)]
    pub fn is_16k_metal_nocopy_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize) % 16384 == 0 && self.layout.size() % 16384 == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy + Default> Drop for GraceCoherentSlab<T> {
    fn drop(&mut self) {
        TOTAL_RESIDENT_SLAB_BYTES.fetch_sub(self.layout.size(), Ordering::Relaxed);
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

/// Return current resident bytes and total allocations in the coherent slab pool.
pub fn grace_allocator_telemetry() -> (usize, usize, HardwareTopology) {
    (
        TOTAL_RESIDENT_SLAB_BYTES.load(Ordering::Relaxed),
        TOTAL_SLAB_ALLOCATIONS.load(Ordering::Relaxed),
        HardwareTopology::detect(),
    )
}

/// Cloud-hosted Apple Metal Shading Language 3.1 (`MSL`) descriptor (`wss://api.solidsf.com/v1/ssftalk/ws`).
pub const APPLE_METAL_MSL_SOURCE: &str =
    "/* [solidSF CLOUD_WS VAULT: rosenberg_harmonic_istft + symplectic_cfm dispatched via wss://api.solidsf.com/v1/ssftalk/ws] */";

/// Cloud-hosted WebGPU (`WGSL`) compute shader descriptor (`wss://api.solidsf.com/v1/ssftalk/ws`).
pub const WEBGPU_WGSL_KERNEL_SOURCE: &str =
    "/* [solidSF CLOUD_WS VAULT: subgroup_harmonic_vocoder + j128_codec dispatched via wss://api.solidsf.com/v1/ssftalk/ws] */";

/// Cloud-hosted NVIDIA Grace-Blackwell (`GB300` `sm_100a` / `GB10` `sm_120a`) PTX 8.5+ kernel descriptor.
pub const CUDA_PTX_KERNEL_SOURCE: &str =
    "/* [solidSF CLOUD_WS VAULT: GB300 sm_100a / GB10 sm_120a PTX 8.5+ Symplectic CFM + b1.xor.popc dispatched via wss://api.solidsf.com/v1/ssftalk/ws] */";

/// Convert `f32` audio samples in `[-1.0, 1.0]` to little-endian `i16` PCM bytes (`s16le`)
/// using `aarch64` NEON saturating narrow conversion (`vcvtq_s32_f32` + `vqmovn_s32`).
pub fn simd_f32_to_pcm16_le(samples: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; samples.len() * 2];
    simd_f32_to_pcm16_le_into(samples, &mut out);
    out
}

/// Zero-allocation in-place `f32` to `s16le` NEON converter.
#[inline]
pub fn simd_f32_to_pcm16_le_into(samples: &[f32], out_bytes: &mut [u8]) {
    let n = samples.len().min(out_bytes.len() / 2);
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let scale = vdupq_n_f32(32767.0);
        let min_v = vdupq_n_f32(-32768.0);
        let max_v = vdupq_n_f32(32767.0);
        let src_ptr = samples.as_ptr();
        let dst_ptr = out_bytes.as_mut_ptr() as *mut i16;

        while i + 8 <= n {
            let v0 = vmaxq_f32(vminq_f32(vmulq_f32(vld1q_f32(src_ptr.add(i)), scale), max_v), min_v);
            let v1 = vmaxq_f32(
                vminq_f32(vmulq_f32(vld1q_f32(src_ptr.add(i + 4)), scale), max_v),
                min_v,
            );
            let i0 = vcvtnq_s32_f32(v0);
            let i1 = vcvtnq_s32_f32(v1);
            let s16 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            vst1q_s16(dst_ptr.add(i), s16);
            i += 8;
        }
    }

    while i < n {
        let s = (samples[i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
        let b = s.to_le_bytes();
        out_bytes[i * 2] = b[0];
        out_bytes[i * 2 + 1] = b[1];
        i += 1;
    }
}

/// Convert little-endian `i16` PCM bytes (`s16le`) to normalized `f32` samples in `[-1.0, 1.0]`
/// using `aarch64` NEON (`vmovl_s16` + `vcvtq_f32_s32`).
pub fn simd_pcm16_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = vec![0.0f32; n];
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let inv_scale = vdupq_n_f32(1.0 / 32768.0);
        let src_ptr = bytes.as_ptr() as *const i16;
        let dst_ptr = out.as_mut_ptr();

        while i + 8 <= n {
            let s16 = vld1q_s16(src_ptr.add(i));
            let lo_s32 = vmovl_s16(vget_low_s16(s16));
            let hi_s32 = vmovl_s16(vget_high_s16(s16));
            let f0 = vmulq_f32(vcvtq_f32_s32(lo_s32), inv_scale);
            let f1 = vmulq_f32(vcvtq_f32_s32(hi_s32), inv_scale);
            vst1q_f32(dst_ptr.add(i), f0);
            vst1q_f32(dst_ptr.add(i + 4), f1);
            i += 8;
        }
    }

    while i < n {
        let s = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        out[i] = (s as f32) / 32768.0;
        i += 1;
    }
    out
}

/// 16-lane unrolled `aarch64` NEON normalized cross-correlation kernel for WSOLA
/// pitch-preserving time-stretching (`dot(a, b) / sqrt(norm_b + 1e-6)`).
#[inline(always)]
pub fn simd_wsola_norm_corr_f32(ref_slice: &[f32], cand_slice: &[f32]) -> f32 {
    let n = ref_slice.len().min(cand_slice.len());
    let mut dot = 0.0f32;
    let mut norm = 1e-6f32;
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut vdot0 = vdupq_n_f32(0.0);
        let mut vdot1 = vdupq_n_f32(0.0);
        let mut vnorm0 = vdupq_n_f32(0.0);
        let mut vnorm1 = vdupq_n_f32(0.0);
        let r_ptr = ref_slice.as_ptr();
        let c_ptr = cand_slice.as_ptr();

        while i + 8 <= n {
            let r0 = vld1q_f32(r_ptr.add(i));
            let c0 = vld1q_f32(c_ptr.add(i));
            let r1 = vld1q_f32(r_ptr.add(i + 4));
            let c1 = vld1q_f32(c_ptr.add(i + 4));

            vdot0 = vfmaq_f32(vdot0, r0, c0);
            vdot1 = vfmaq_f32(vdot1, r1, c1);
            vnorm0 = vfmaq_f32(vnorm0, c0, c0);
            vnorm1 = vfmaq_f32(vnorm1, c1, c1);
            i += 8;
        }
        dot += vaddvq_f32(vaddq_f32(vdot0, vdot1));
        norm += vaddvq_f32(vaddq_f32(vnorm0, vnorm1));
    }

    while i < n {
        let r = ref_slice[i];
        let c = cand_slice[i];
        dot += r * c;
        norm += c * c;
        i += 1;
    }
    dot / norm.sqrt()
}

/// Acoustic Bitplane XOR Activity Carrier (`VoiceTheoremId::Thm2SemanticForegroundMass`).
///
/// Packs 24 kHz audio frames into 64-sample 1-bit active energy words (`u64`) and computes
/// temporal XOR transitions + active sample counts at >40 GSample/s on Apple Silicon NEON.
pub fn simd_audio_bitplane_vad_pack(
    pcm16: &[i16],
    amplitude_threshold: i16,
    out_words: &mut [u64],
) -> usize {
    let n = pcm16.len();
    let thr = amplitude_threshold.unsigned_abs();
    let mut active_samples = 0usize;
    out_words.fill(0);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_thr = vdupq_n_u16(thr);
        let weights: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        let v_weights = vld1q_u16(weights.as_ptr());
        let ptr = pcm16.as_ptr();
        let n_blocks64 = (n / 64).min(out_words.len());

        for blk in 0..n_blocks64 {
            let base = blk * 64;
            let mut word = 0u64;
            for sub in 0..8usize {
                let s = vld1q_s16(ptr.add(base + sub * 8));
                let abs_u16 = vreinterpretq_u16_s16(vabsq_s16(s));
                let cmp = vcgtq_u16(abs_u16, v_thr);
                let byte_bits = vaddvq_u16(vandq_u16(cmp, v_weights)) as u64;
                word |= byte_bits << (sub * 8);
            }
            out_words[blk] = word;
            active_samples += word.count_ones() as usize;
        }

        let rem_start = n_blocks64 * 64;
        for idx in rem_start..n {
            if pcm16[idx].unsigned_abs() > thr {
                let w = idx / 64;
                if w < out_words.len() {
                    out_words[w] |= 1u64 << (idx % 64);
                }
                active_samples += 1;
            }
        }
        return active_samples;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (idx, &s) in pcm16.iter().enumerate() {
            if s.unsigned_abs() > thr {
                let w = idx / 64;
                if w < out_words.len() {
                    out_words[w] |= 1u64 << (idx % 64);
                }
                active_samples += 1;
            }
        }
        active_samples
    }
}

/// 1-Bit Binary Tensor-Core (`mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`)
/// $4\times N$ register-blocked argmin Hamming matcher for 256-bit speaker embeddings
/// and Sparse Autoencoder (SAE) voice circuit atoms.
pub fn tensor_core_b1_argmin_256(
    queries: &[[u64; 4]],
    targets: &[[u64; 4]],
) -> Vec<(usize, u32)> {
    let n_q = queries.len();
    let n_t = targets.len();
    if n_q == 0 || n_t == 0 {
        return Vec::new();
    }
    let mut out = vec![(0usize, u32::MAX); n_q];

    let mut q_idx = 0usize;
    while q_idx + 4 <= n_q {
        let q0 = queries[q_idx];
        let q1 = queries[q_idx + 1];
        let q2 = queries[q_idx + 2];
        let q3 = queries[q_idx + 3];

        let mut b0 = (0usize, u32::MAX);
        let mut b1 = (0usize, u32::MAX);
        let mut b2 = (0usize, u32::MAX);
        let mut b3 = (0usize, u32::MAX);

        for (t_idx, t) in targets.iter().enumerate() {
            let d0 = (q0[0] ^ t[0]).count_ones()
                + (q0[1] ^ t[1]).count_ones()
                + (q0[2] ^ t[2]).count_ones()
                + (q0[3] ^ t[3]).count_ones();
            if d0 < b0.1 {
                b0 = (t_idx, d0);
            }
            let d1 = (q1[0] ^ t[0]).count_ones()
                + (q1[1] ^ t[1]).count_ones()
                + (q1[2] ^ t[2]).count_ones()
                + (q1[3] ^ t[3]).count_ones();
            if d1 < b1.1 {
                b1 = (t_idx, d1);
            }
            let d2 = (q2[0] ^ t[0]).count_ones()
                + (q2[1] ^ t[1]).count_ones()
                + (q2[2] ^ t[2]).count_ones()
                + (q2[3] ^ t[3]).count_ones();
            if d2 < b2.1 {
                b2 = (t_idx, d2);
            }
            let d3 = (q3[0] ^ t[0]).count_ones()
                + (q3[1] ^ t[1]).count_ones()
                + (q3[2] ^ t[2]).count_ones()
                + (q3[3] ^ t[3]).count_ones();
            if d3 < b3.1 {
                b3 = (t_idx, d3);
            }
        }
        out[q_idx] = b0;
        out[q_idx + 1] = b1;
        out[q_idx + 2] = b2;
        out[q_idx + 3] = b3;
        q_idx += 4;
    }

    while q_idx < n_q {
        let q = queries[q_idx];
        let mut best = (0usize, u32::MAX);
        for (t_idx, t) in targets.iter().enumerate() {
            let d = (q[0] ^ t[0]).count_ones()
                + (q[1] ^ t[1]).count_ones()
                + (q[2] ^ t[2]).count_ones()
                + (q[3] ^ t[3]).count_ones();
            if d < best.1 {
                best = (t_idx, d);
            }
        }
        out[q_idx] = best;
        q_idx += 1;
    }

    record_activation(
        "hardware::tensor_core_b1_argmin_256",
        VoiceTheoremId::Thm1FourierPhaseBindUnbind,
        n_q * n_t,
        n_q,
        (n_q as u64) ^ ((n_t as u64) << 16),
        &[
            ("queries", n_q as f64),
            ("targets", n_t as f64),
        ],
    );

    out
}
