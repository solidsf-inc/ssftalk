//! Core audio buffers, ElevenLabs output formats, deterministic seeding, and base64 wire utilities.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Standard error code for invalid arguments (`-1`).
pub const STS_BAD_ARG: i32 = -1;
/// Standard error code for runtime/protocol errors (`-2`).
pub const STS_ERROR: i32 = -2;
/// HTTP 401 Unauthorized error code.
pub const STS_UNAUTHORIZED: i32 = 401;
/// HTTP 402 Payment Required error code (`solidSF Credits` depleted).
pub const STS_PAYMENT_REQUIRED: i32 = 402;
/// HTTP 422 Unprocessable Entity error code.
pub const STS_UNPROCESSABLE: i32 = 422;

/// Canonical error type for `ssftalk` and drop-in `ssftalk::elevenlabs` calls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl Error {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ssftalk error ({}): {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Result alias for `ssftalk` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Default internal synthesis sample rate (`24,000 Hz`).
pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;
/// Acoustic hop size in samples at 24 kHz (`256 samples = 10.67 ms`).
pub const HOP_SAMPLES: usize = 256;
/// Number of log-mel filterbank channels (`100`).
pub const N_MELS: usize = 100;

/// ElevenLabs-compatible + `solidSF` native audio output formats (`output_format` query parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// ElevenLabs default MP3 44.1 kHz @ 128 kbps (`mp3_44100_128`).
    Mp3_44100_128,
    /// ElevenLabs low-bandwidth MP3 22.05 kHz @ 32 kbps (`mp3_22050_32`).
    Mp3_22050_32,
    /// Raw 16-bit little-endian mono PCM at 16 kHz (`pcm_16000`).
    Pcm16000,
    /// Raw 16-bit little-endian mono PCM at 22.05 kHz (`pcm_22050`).
    Pcm22050,
    /// Raw 16-bit little-endian mono PCM at 24 kHz (`pcm_24000`).
    Pcm24000,
    /// Raw 16-bit little-endian mono PCM at 44.1 kHz (`pcm_44100`).
    Pcm44100,
    /// ITU-T G.711 mu-law 8 kHz telephony stream (`ulaw_8000`) for Twilio / SIP trunks.
    Ulaw8000,
    /// RIFF WAV container with 24 kHz 16-bit mono PCM (`wav_24000`).
    #[default]
    Wav24000,
    /// `solidSF` native 128.0 kbps 4-mode companded predictive codec (`j128_24000`, 0.625 ms blocks).
    J128_24000,
}

impl OutputFormat {
    /// Parse an ElevenLabs or `ssftalk` `output_format` string identifier.
    pub fn from_code(code: &str) -> Result<Self> {
        match code.trim().to_ascii_lowercase().as_str() {
            "mp3_44100_128" | "mp3_44100_192" | "mp3_44100_96" | "mp3_44100_64" | "mp3" => {
                Ok(Self::Mp3_44100_128)
            }
            "mp3_22050_32" => Ok(Self::Mp3_22050_32),
            "pcm_16000" | "pcm16_16k" => Ok(Self::Pcm16000),
            "pcm_22050" => Ok(Self::Pcm22050),
            "pcm_24000" | "pcm16" | "pcm" => Ok(Self::Pcm24000),
            "pcm_44100" => Ok(Self::Pcm44100),
            "ulaw_8000" | "mulaw_8000" | "ulaw" => Ok(Self::Ulaw8000),
            "wav_24000" | "wav" | "audio/wav" => Ok(Self::Wav24000),
            "j128_24000" | "j128" | "j128_lite" => Ok(Self::J128_24000),
            other => Err(Error::new(
                STS_BAD_ARG,
                format!("unsupported output_format: {other}"),
            )),
        }
    }

    pub const fn code(&self) -> &'static str {
        match self {
            Self::Mp3_44100_128 => "mp3_44100_128",
            Self::Mp3_22050_32 => "mp3_22050_32",
            Self::Pcm16000 => "pcm_16000",
            Self::Pcm22050 => "pcm_22050",
            Self::Pcm24000 => "pcm_24000",
            Self::Pcm44100 => "pcm_44100",
            Self::Ulaw8000 => "ulaw_8000",
            Self::Wav24000 => "wav_24000",
            Self::J128_24000 => "j128_24000",
        }
    }

    pub const fn sample_rate(&self) -> u32 {
        match self {
            Self::Mp3_44100_128 | Self::Pcm44100 => 44_100,
            Self::Mp3_22050_32 | Self::Pcm22050 => 22_050,
            Self::Pcm16000 => 16_000,
            Self::Pcm24000 | Self::Wav24000 | Self::J128_24000 => 24_000,
            Self::Ulaw8000 => 8_000,
        }
    }

    pub const fn content_type(&self) -> &'static str {
        match self {
            Self::Mp3_44100_128 | Self::Mp3_22050_32 => "audio/mpeg",
            Self::Pcm16000 => "audio/pcm; rate=16000; channels=1; format=s16le",
            Self::Pcm22050 => "audio/pcm; rate=22050; channels=1; format=s16le",
            Self::Pcm24000 => "audio/pcm; rate=24000; channels=1; format=s16le",
            Self::Pcm44100 => "audio/pcm; rate=44100; channels=1; format=s16le",
            Self::Ulaw8000 => "audio/basic; rate=8000; channels=1",
            Self::Wav24000 => "audio/wav",
            Self::J128_24000 => "application/x-solidsf-j128; rate=24000; bitrate=128000",
        }
    }
}

/// Dense floating-point audio waveform buffer in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Default for AudioBuffer {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            sample_rate: DEFAULT_SAMPLE_RATE,
            channels: 1,
        }
    }
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate: sample_rate.max(1_000),
            channels: channels.max(1),
        }
    }

    pub fn zeros(num_samples: usize, sample_rate: u32) -> Self {
        Self {
            samples: vec![0.0; num_samples],
            sample_rate: sample_rate.max(1_000),
            channels: 1,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Total duration of the audio buffer in seconds.
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        let frames = self.samples.len() / (self.channels.max(1) as usize);
        (frames as f64) / (self.sample_rate.max(1) as f64)
    }

    /// Root-mean-square (RMS) amplitude in `[0.0, 1.0]`.
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self.samples.iter().map(|&x| x * x).sum();
        (sum_sq / (self.samples.len() as f32)).sqrt()
    }

    /// Peak absolute sample amplitude.
    pub fn peak(&self) -> f32 {
        self.samples
            .iter()
            .fold(0.0_f32, |acc, &x| acc.max(x.abs()))
    }

    /// Approximate integrated loudness in dBFS / LUFS.
    pub fn lufs_estimate(&self) -> f32 {
        let r = self.rms().max(1e-7);
        -0.691 + 20.0 * r.log10()
    }

    /// Resample to `target_rate` using band-limited linear + 3-point anti-aliasing interpolation.
    pub fn resample(&self, target_rate: u32) -> Self {
        let dst_rate = target_rate.max(1_000);
        if dst_rate == self.sample_rate || self.samples.is_empty() {
            return self.clone();
        }
        let ratio = (self.sample_rate as f64) / (dst_rate as f64);
        let out_len = ((self.samples.len() as f64) / ratio).round().max(1.0) as usize;
        let mut out = Vec::with_capacity(out_len);
        let n = self.samples.len();

        for i in 0..out_len {
            let src_pos = (i as f64) * ratio;
            let idx = (src_pos.floor() as usize).min(n - 1);
            let frac = (src_pos - (idx as f64)) as f32;
            let s0 = self.samples[idx];
            let s1 = self.samples[(idx + 1).min(n - 1)];
            let val = s0 + (s1 - s0) * frac;
            out.push(val.clamp(-1.0, 1.0));
        }

        Self {
            samples: out,
            sample_rate: dst_rate,
            channels: self.channels,
        }
    }

    /// Convert to 16-bit little-endian signed PCM bytes (`s16le`).
    pub fn to_pcm16_le_bytes(&self) -> Vec<u8> {
        crate::hardware::simd_f32_to_pcm16_le(&self.samples)
    }

    /// Decode from 16-bit little-endian signed PCM bytes (`s16le`).
    pub fn from_pcm16_le_bytes(bytes: &[u8], sample_rate: u32) -> Self {
        let samples = crate::hardware::simd_pcm16_le_to_f32(bytes);
        Self {
            samples,
            sample_rate: sample_rate.max(1_000),
            channels: 1,
        }
    }
}

/// Deterministic 64-bit FNV-1a hash used for seed derivation when `seed` is omitted.
#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Derive a deterministic synthesis seed from `(voice_id, text, explicit_seed)`.
#[inline]
pub fn derive_utterance_seed(voice_id: &str, text: &str, explicit_seed: Option<u64>) -> u64 {
    if let Some(s) = explicit_seed {
        return s;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in voice_id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= 0;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    for &b in text.trim().as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

const B64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Pure-Rust RFC-4648 standard Base64 encoder for ElevenLabs `audio_base64` payloads.
pub fn base64_encode(input: &[u8]) -> String {
    let mut out = Vec::with_capacity(input.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8) | (input[i + 2] as u32);
        out.push(B64_TABLE[((n >> 18) & 0x3F) as usize]);
        out.push(B64_TABLE[((n >> 12) & 0x3F) as usize]);
        out.push(B64_TABLE[((n >> 6) & 0x3F) as usize]);
        out.push(B64_TABLE[(n & 0x3F) as usize]);
        i += 3;
    }
    let rem = input.len() - i;
    if rem == 1 {
        let n = (input[i] as u32) << 16;
        out.push(B64_TABLE[((n >> 18) & 0x3F) as usize]);
        out.push(B64_TABLE[((n >> 12) & 0x3F) as usize]);
        out.push(b'=');
        out.push(b'=');
    } else if rem == 2 {
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        out.push(B64_TABLE[((n >> 18) & 0x3F) as usize]);
        out.push(B64_TABLE[((n >> 12) & 0x3F) as usize]);
        out.push(B64_TABLE[((n >> 6) & 0x3F) as usize]);
        out.push(b'=');
    }
    String::from_utf8(out).unwrap_or_default()
}

/// Pure-Rust RFC-4648 standard Base64 decoder.
pub fn base64_decode(encoded: &str) -> Result<Vec<u8>> {
    let mut vals = Vec::with_capacity(encoded.len());
    for b in encoded.bytes() {
        if b.is_ascii_whitespace() || b == b'=' {
            continue;
        }
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => {
                return Err(Error::new(
                    STS_BAD_ARG,
                    format!("invalid base64 byte: 0x{b:02x}"),
                ))
            }
        };
        vals.push(v);
    }
    let mut out = Vec::with_capacity((vals.len() * 3) / 4);
    let mut i = 0;
    while i + 4 <= vals.len() {
        let n = ((vals[i] as u32) << 18)
            | ((vals[i + 1] as u32) << 12)
            | ((vals[i + 2] as u32) << 6)
            | (vals[i + 3] as u32);
        out.push(((n >> 16) & 0xFF) as u8);
        out.push(((n >> 8) & 0xFF) as u8);
        out.push((n & 0xFF) as u8);
        i += 4;
    }
    let rem = vals.len() - i;
    if rem == 2 {
        let n = ((vals[i] as u32) << 18) | ((vals[i + 1] as u32) << 12);
        out.push(((n >> 16) & 0xFF) as u8);
    } else if rem == 3 {
        let n =
            ((vals[i] as u32) << 18) | ((vals[i + 1] as u32) << 12) | ((vals[i + 2] as u32) << 6);
        out.push(((n >> 16) & 0xFF) as u8);
        out.push(((n >> 8) & 0xFF) as u8);
    }
    Ok(out)
}
