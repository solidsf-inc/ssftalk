//! `studio`: Pure-Rust AI Audio Studio, BGM & Structured SFX Synthesis, Audio-to-Audio Editing
//! (`variation` / `inpaint` / `continue`), EBU R128 Loudness Mastering, Seamless Loop Crossfader,
//! 24-Bit WAV / OGG / FLAC Exporter, Reproducible `.json` Sidecar Lineage, and MCP Server (`solidSF`).
//!
//! Folds all desirable capabilities of local/cloud AI Audio Studio workflows into `ssftalk`:
//! 1. **`generate_audio`**: Text-to-BGM (`music`), solo stems/loops (`instrument`), and 4-field
//!    structured physical sound effects (`sfx`: `sound_source`, `sound_action`, `sound_material`,
//!    `sound_space`), `1..=4` seed variations per call, `bpm`, `cfg`, `steps`, `negative_prompt`,
//!    and `-1.0 dBFS` safety peak ceiling.
//! 2. **`edit_audio`**: Non-destructive audio editing across three modes:
//!    - `variation`: Audio-to-audio style/timbre transfer controlled by `strength` (`0.05..=1.0`).
//!    - `inpaint`: Regenerate a `[start_seconds, end_seconds]` window while keeping surrounding context.
//!    - `continue`: Extend a track to `duration_seconds`, automatically detecting `audible_end`
//!      (last 5 ms RMS window louder than `-60 dBFS`) and backing up by `overlap_seconds` (`1.0s`)
//!      so trailing fades or silence are replaced with a smooth equal-power join.
//! 3. **`process_audio`**: Non-destructive 8-stage mastering and game/video export pipeline:
//!    `trim` $\to$ `trim_silence` $\to$ `mono` $\to$ `resample` (`44.1 kHz` / `48 kHz`) + two-pass
//!    EBU R128 `loudnorm` (`-14` streaming, `-16` game BGM, `-23` broadcast LUFS) $\to$ equal-power
//!    seamless `loop` crossfade $\to$ $\sin^2 / \cos^2$ `fade` $\to$ `peak_normalize` / `peak_ceiling`
//!    $\to$ 24-bit `wav` / `ogg` / `flac` + `.json` metadata sidecar preserving upstream `source` lineage.
//! 4. **`unload_models` & Lazy Slab Lifecycle**: Length-agnostic model caching (`medium + same-l`
//!    up to `380s`, `sm-music + same-s` and `sm-sfx + same-s` up to `120s`), UMA buffer cache cap,
//!    idle timeout reaper, explicit `unload_models`, MCP JSON-RPC 2.0 tool server, and `SSFTK\x01`
//!    Cloud WebSocket billing integration.

use crate::cloud_ws::{SsfTalkWsClient, WsEngine, WsRpcRequest, WsRpcResponse};
use crate::core::{fnv1a64, AudioBuffer, Error, Result, STS_BAD_ARG, STS_ERROR};
use crate::mechinterp::{record_activation, VoiceTheoremId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Default studio generation sample rate (`44,100 Hz` stereo).
pub const STUDIO_DEFAULT_SAMPLE_RATE: u32 = 44_100;
/// Standard video/broadcast mastering sample rate (`48,000 Hz`).
pub const STUDIO_VIDEO_SAMPLE_RATE: u32 = 48_000;

/// Built-in MCP server instructions and Stable Audio 3 / `ssftalk::studio` prompt engineering guide.
pub const STUDIO_MCP_INSTRUCTIONS: &str = "\
solidSF ssftalk Audio Studio (local Apple Silicon UMA / NVIDIA Tensor-Core + Cloud WebSocket) for BGM, instrument stems, and sound effects.
generate_audio/edit_audio write 44.1 kHz 24-bit stereo WAV + a .json sidecar (prompt, seed, settings) next to each file;
process_audio trims, loops, fades, normalizes (peak or EBU R128 LUFS) and exports 48 kHz / 44.1 kHz WAV, OGG Vorbis, or 24-bit FLAC for game engines and video editors.

Prompting rules:
- Pick audio_type; the server adds the canonical prefix:
  music -> \"TrackType: Music, VocalType: Instrumental, \"
  instrument (solo/stem/loop) -> \"TrackType: Instrument, \"
  sfx (sound effects, ambience, one-shots) -> \"TrackType: SFX, \"
  raw -> no prefix (prompt used verbatim).
- Music: genre/style + adjectives -> lead instruments -> supporting layers/FX -> drums/rhythm -> mood/energy. Pass tempo as bpm.
- SFX: fill sound_source (what makes it), sound_action (how triggered and evolves: attack, decay, movement), sound_material (texture, timbre, material), and sound_space (room, distance, dry/reverberant, mic).
- Match duration to content: impacts 1-3 s, actions 3-6 s, ambience 6-15 s+, loops 6-20 s, full tracks 120-300 s (max 380 s on medium, 120 s on sm-music / sm-sfx).
- Defaults steps=8, cfg=1.0; raise cfg to 3-7 for stricter prompt adherence or negative_prompt.
- Generated files receive a safety peak ceiling (-1.0 dBFS default: louder peaks are turned down, quieter audio is never boosted). Use process_audio to normalize to peak or target LUFS (-14 streaming/video, -16 game BGM, -23 broadcast).";

/// Studio audio generation category (`audio_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StudioAudioType {
    #[default]
    Music,
    Instrument,
    Sfx,
    Raw,
}

impl StudioAudioType {
    pub fn from_str_lossy(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "music" | "bgm" => Ok(Self::Music),
            "instrument" | "stem" | "loop" => Ok(Self::Instrument),
            "sfx" | "sound_effect" | "fx" => Ok(Self::Sfx),
            "raw" => Ok(Self::Raw),
            other => Err(Error::new(
                STS_BAD_ARG,
                format!("invalid audio_type {other:?}: expected music, instrument, sfx, or raw"),
            )),
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Music => "music",
            Self::Instrument => "instrument",
            Self::Sfx => "sfx",
            Self::Raw => "raw",
        }
    }

    /// Official training `TrackType` prefix.
    pub const fn prefix(&self) -> &'static str {
        match self {
            Self::Music => "TrackType: Music, VocalType: Instrumental, ",
            Self::Instrument => "TrackType: Instrument, ",
            Self::Sfx => "TrackType: SFX, ",
            Self::Raw => "",
        }
    }

    /// Default duration in seconds when `duration_seconds` and `"Length: N seconds"` are omitted.
    pub const fn default_seconds(&self) -> f64 {
        match self {
            Self::Music => 60.0,
            Self::Instrument => 12.0,
            Self::Sfx => 5.0,
            Self::Raw => 30.0,
        }
    }
}

/// Structured 4-field sound-effect descriptor (`sound_source`, `sound_action`, `sound_material`, `sound_space`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SfxSoundFields {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub material: String,
    #[serde(default)]
    pub space: String,
}

impl SfxSoundFields {
    pub fn new(
        source: impl Into<String>,
        action: impl Into<String>,
        material: impl Into<String>,
        space: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            action: action.into(),
            material: material.into(),
            space: space.into(),
        }
    }

    pub fn as_array(&self) -> [&str; 4] {
        [
            self.source.as_str(),
            self.action.as_str(),
            self.material.as_str(),
            self.space.as_str(),
        ]
    }
}

/// Parse a trailing `"Length: N seconds"` suffix (case-insensitive) from `text`,
/// returning `(text_without_suffix, parsed_seconds)`.
fn extract_trailing_length_seconds(text: &str) -> (&str, Option<f64>) {
    let trimmed = text.trim_end();
    let no_trail_dot = trimmed.strip_suffix('.').unwrap_or(trimmed).trim_end();
    let lower = no_trail_dot.to_ascii_lowercase();
    let without_unit = if let Some(s) = lower.strip_suffix("seconds") {
        s.trim_end()
    } else if let Some(s) = lower.strip_suffix("second") {
        s.trim_end()
    } else {
        return (trimmed, None);
    };

    // Find "length:" before the numeric value
    let Some(colon_pos) = without_unit.rfind("length:") else {
        return (trimmed, None);
    };
    let num_str = without_unit[colon_pos + "length:".len()..].trim();
    if num_str.is_empty() || !num_str.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return (trimmed, None);
    }
    let Ok(val) = num_str.parse::<f64>() else {
        return (trimmed, None);
    };

    let prefix_part = no_trail_dot[..colon_pos].trim_end();
    let clean_prefix = prefix_part.strip_suffix('.').unwrap_or(prefix_part).trim_end();
    (clean_prefix, Some(val))
}

fn contains_bpm_word(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    for token in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        if token == "bpm" {
            return true;
        }
    }
    false
}

/// Build the structured prompt the synthesis model receives:
/// `prompt [, source, action, material, space]. [BPM: N.]`
/// with the canonical `TrackType` prefix, and pull a trailing `"Length: N seconds"`
/// out of the prompt text into the duration when `seconds` is `None`.
pub fn prepare_studio_prompt(
    prompt: &str,
    audio_type: StudioAudioType,
    seconds: Option<f64>,
    bpm: Option<u32>,
    sfx_fields: &SfxSoundFields,
) -> Result<(String, Option<f64>)> {
    let trimmed = prompt.trim();
    let (base_text, parsed_len) = extract_trailing_length_seconds(trimmed);
    let resolved_seconds = seconds.or(parsed_len);

    let mut parts: Vec<String> = Vec::with_capacity(5);
    for raw in std::iter::once(base_text).chain(sfx_fields.as_array().into_iter()) {
        let p = raw.trim().trim_end_matches([' ', '.', ',']);
        if !p.is_empty() {
            parts.push(p.to_string());
        }
    }
    if parts.is_empty() {
        return Err(Error::new(
            STS_BAD_ARG,
            "empty prompt: give prompt and/or the sound_* fields",
        ));
    }

    let mut text = format!("{}.", parts.join(", "));
    if let Some(b) = bpm {
        if b > 0 && !contains_bpm_word(&text) {
            text.push_str(&format!(" BPM: {b}."));
        }
    }
    if audio_type != StudioAudioType::Raw
        && !text.to_ascii_lowercase().starts_with("tracktype:")
    {
        text = format!("{}{text}", audio_type.prefix());
    }
    Ok((text, resolved_seconds))
}

/// Drop the `TrackType: ..., (VocalType: ..., )?` prefix (used for auto file names).
pub fn strip_studio_prefix(prompt: &str) -> String {
    let trimmed = prompt.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("tracktype:") {
        return trimmed.to_string();
    }
    let Some(first_comma) = trimmed.find(',') else {
        return trimmed.to_string();
    };
    let after_first = trimmed[first_comma + 1..].trim_start();
    if after_first.to_ascii_lowercase().starts_with("vocaltype:") {
        if let Some(second_comma) = after_first.find(',') {
            return after_first[second_comma + 1..].trim_start().to_string();
        }
    }
    after_first.to_string()
}

/// Convert decibels to linear amplitude (`10^(db / 20)`).
#[inline]
pub fn db_to_lin(db: f64) -> f32 {
    10.0_f64.powf(db / 20.0) as f32
}

/// Convert linear amplitude to decibels (`20 * log10(max(x, 1e-12))`).
#[inline]
pub fn lin_to_db(x: f32) -> f64 {
    20.0 * (x.max(1e-12) as f64).log10()
}

#[inline]
fn round_to(val: f64, decimals: u32) -> f64 {
    let factor = 10.0_f64.powi(decimals as i32);
    (val * factor).round() / factor
}

/// Summary amplitude telemetry for a studio buffer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioSignalStats {
    pub peak: f64,
    pub peak_dbfs: f64,
    pub rms: f64,
}

/// Two-pass EBU R128 / ITU-R BS.1770 loudness normalization telemetry.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LoudnormTelemetry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_lufs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_true_peak_db: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_lufs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loudnorm_mode: Option<String>,
}

/// Export container format for `process_audio` and studio file I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StudioExportFormat {
    /// 24-bit stereo/mono PCM RIFF WAVE (`.wav`).
    #[default]
    Wav,
    /// 16-bit stereo/mono PCM RIFF WAVE (`.wav`).
    Wav16,
    /// Ogg Vorbis container (`.ogg`) for Unity / Godot / Unreal game audio.
    Ogg,
    /// 24-bit lossless FLAC container (`.flac`).
    Flac,
}

impl StudioExportFormat {
    pub fn from_str_lossy(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "wav" | "wav24" | "pcm_24" => Ok(Self::Wav),
            "wav16" | "pcm_16" => Ok(Self::Wav16),
            "ogg" | "vorbis" | "ogg_vorbis" => Ok(Self::Ogg),
            "flac" | "flac24" => Ok(Self::Flac),
            other => Err(Error::new(
                STS_BAD_ARG,
                format!("unknown studio export format {other:?}: expected wav, ogg, or flac"),
            )),
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Wav16 => "wav16",
            Self::Ogg => "ogg",
            Self::Flac => "flac",
        }
    }

    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Wav | Self::Wav16 => ".wav",
            Self::Ogg => ".ogg",
            Self::Flac => ".flac",
        }
    }

    pub const fn content_type(&self) -> &'static str {
        match self {
            Self::Wav | Self::Wav16 => "audio/wav",
            Self::Ogg => "audio/ogg",
            Self::Flac => "audio/flac",
        }
    }
}

/// Multi-channel planar float32 studio waveform (`channels[ch][sample]`), matching `(channels, samples)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudioAudioClip {
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

impl StudioAudioClip {
    pub fn new(channels: Vec<Vec<f32>>, sample_rate: u32) -> Result<Self> {
        if channels.is_empty() {
            return Err(Error::new(STS_BAD_ARG, "StudioAudioClip requires at least 1 channel"));
        }
        let len = channels[0].len();
        for ch in &channels {
            if ch.len() != len {
                return Err(Error::new(
                    STS_BAD_ARG,
                    "all channels in StudioAudioClip must have identical sample length",
                ));
            }
        }
        Ok(Self {
            channels,
            sample_rate: sample_rate.max(1_000),
        })
    }

    pub fn zeros(num_channels: usize, num_samples: usize, sample_rate: u32) -> Self {
        let ch = num_channels.max(1);
        Self {
            channels: vec![vec![0.0; num_samples]; ch],
            sample_rate: sample_rate.max(1_000),
        }
    }

    pub fn from_mono_buffer(buf: &AudioBuffer) -> Self {
        Self {
            channels: vec![buf.samples.clone()],
            sample_rate: buf.sample_rate,
        }
    }

    pub fn to_mono_buffer(&self) -> AudioBuffer {
        let mono = self.to_mono();
        AudioBuffer::new(
            mono.channels.into_iter().next().unwrap_or_default(),
            self.sample_rate,
            1,
        )
    }

    #[inline]
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    #[inline]
    pub fn num_samples(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        (self.num_samples() as f64) / (self.sample_rate.max(1) as f64)
    }

    /// Compute peak, peak dBFS, and RMS statistics across all channels and samples.
    pub fn stats(&self) -> AudioSignalStats {
        let total = self.num_channels() * self.num_samples();
        if total == 0 {
            return AudioSignalStats {
                peak: 0.0,
                peak_dbfs: round_to(lin_to_db(0.0), 2),
                rms: 0.0,
            };
        }
        let mut peak = 0.0_f32;
        let mut sum_sq = 0.0_f64;
        for ch in &self.channels {
            for &s in ch {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
                sum_sq += (s as f64) * (s as f64);
            }
        }
        let rms = (sum_sq / (total as f64)).sqrt();
        AudioSignalStats {
            peak: round_to(peak as f64, 3),
            peak_dbfs: round_to(lin_to_db(peak), 2),
            rms: round_to(rms, 4),
        }
    }

    /// Safety peak ceiling: attenuates the clip if its peak exceeds `ceiling_db`. Never boosts.
    /// Returns `(clip, applied_gain_db)`.
    pub fn peak_ceiling(&self, ceiling_db: f64) -> (Self, f64) {
        let mut peak = 0.0_f32;
        for ch in &self.channels {
            for &s in ch {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
            }
        }
        let limit = db_to_lin(ceiling_db);
        if peak <= limit || peak <= 0.0 {
            return (self.clone(), 0.0);
        }
        let scale = limit / peak;
        let gain_db = lin_to_db(scale);
        let scaled = self
            .channels
            .iter()
            .map(|ch| ch.iter().map(|&s| s * scale).collect())
            .collect();
        (
            Self {
                channels: scaled,
                sample_rate: self.sample_rate,
            },
            gain_db,
        )
    }

    /// Peak normalize: scales the clip (up or down) so its peak hits `target_db`.
    /// Returns `(clip, applied_gain_db)`.
    pub fn peak_normalize(&self, target_db: f64) -> (Self, f64) {
        let mut peak = 0.0_f32;
        for ch in &self.channels {
            for &s in ch {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
            }
        }
        if peak <= 0.0 {
            return (self.clone(), 0.0);
        }
        let scale = db_to_lin(target_db) / peak;
        let gain_db = lin_to_db(scale);
        let scaled = self
            .channels
            .iter()
            .map(|ch| ch.iter().map(|&s| s * scale).collect())
            .collect();
        (
            Self {
                channels: scaled,
                sample_rate: self.sample_rate,
            },
            gain_db,
        )
    }

    /// Trim to `[start_seconds, end_seconds)`.
    pub fn trim(&self, start_seconds: f64, end_seconds: Option<f64>) -> Result<Self> {
        let len = self.num_samples();
        let sr = self.sample_rate as f64;
        let s = (start_seconds.max(0.0) * sr).round() as usize;
        let e = match end_seconds {
            Some(end) => ((end.max(0.0) * sr).round() as usize).min(len),
            None => len,
        };
        if e <= s {
            return Err(Error::new(
                STS_BAD_ARG,
                format!("empty trim range {start_seconds}-{end_seconds:?}s"),
            ));
        }
        let trimmed = self.channels.iter().map(|ch| ch[s..e].to_vec()).collect();
        Ok(Self {
            channels: trimmed,
            sample_rate: self.sample_rate,
        })
    }

    /// Seconds up to the end of the last 5 ms RMS window louder than `threshold_db` (default `-60.0 dB`).
    pub fn audible_end(&self, threshold_db: f64) -> f64 {
        let len = self.num_samples();
        let sr = self.sample_rate as usize;
        let win = ((sr as f64 * 0.005) as usize).max(1);
        let n_wins = len / win;
        if n_wins == 0 {
            return (len as f64) / (self.sample_rate as f64);
        }
        let thr = db_to_lin(threshold_db) as f64;
        let mut last_loud: Option<usize> = None;
        for w in 0..n_wins {
            let base = w * win;
            let mut sum_sq = 0.0_f64;
            for i in 0..win {
                let mut m = 0.0_f32;
                for ch in &self.channels {
                    let a = ch[base + i].abs();
                    if a > m {
                        m = a;
                    }
                }
                sum_sq += (m as f64) * (m as f64);
            }
            let rms = (sum_sq / (win as f64)).sqrt();
            if rms > thr {
                last_loud = Some(w);
            }
        }
        match last_loud {
            Some(w) => (((w + 1) * win).min(len) as f64) / (self.sample_rate as f64),
            None => 0.0,
        }
    }

    /// Cut leading and trailing silence quieter than `threshold_db` (5 ms RMS windows) with `pad_ms` margin.
    pub fn trim_silence(&self, threshold_db: f64, pad_ms: f64) -> Self {
        let len = self.num_samples();
        let sr = self.sample_rate as usize;
        let win = ((sr as f64 * 0.005) as usize).max(1);
        let n_wins = len / win;
        if n_wins == 0 {
            return self.clone();
        }
        let thr = db_to_lin(threshold_db) as f64;
        let mut first_loud: Option<usize> = None;
        let mut last_loud: Option<usize> = None;

        for w in 0..n_wins {
            let base = w * win;
            let mut sum_sq = 0.0_f64;
            for i in 0..win {
                let mut m = 0.0_f32;
                for ch in &self.channels {
                    let a = ch[base + i].abs();
                    if a > m {
                        m = a;
                    }
                }
                sum_sq += (m as f64) * (m as f64);
            }
            let rms = (sum_sq / (win as f64)).sqrt();
            if rms > thr {
                if first_loud.is_none() {
                    first_loud = Some(w);
                }
                last_loud = Some(w);
            }
        }

        let (Some(first), Some(last)) = (first_loud, last_loud) else {
            return self.clone();
        };
        let pad = ((self.sample_rate as f64) * pad_ms / 1000.0) as usize;
        let s = (first * win).saturating_sub(pad);
        let e = ((last + 1) * win + pad).min(len);
        let sliced = self.channels.iter().map(|ch| ch[s..e].to_vec()).collect();
        Self {
            channels: sliced,
            sample_rate: self.sample_rate,
        }
    }

    /// Apply equal-power $\sin^2$ fade-in and $\cos^2$ fade-out envelopes.
    pub fn fade(&self, fade_in_ms: f64, fade_out_ms: f64) -> Self {
        let mut out = self.clone();
        let len = out.num_samples();
        if len == 0 {
            return out;
        }
        let n_in = (((self.sample_rate as f64) * fade_in_ms / 1000.0) as usize).min(len);
        let n_out = (((self.sample_rate as f64) * fade_out_ms / 1000.0) as usize).min(len);

        if n_in > 0 {
            let denom = (n_in.saturating_sub(1)).max(1) as f64;
            for i in 0..n_in {
                let t = std::f64::consts::FRAC_PI_2 * (i as f64) / denom;
                let w = (t.sin() * t.sin()) as f32;
                for ch in &mut out.channels {
                    ch[i] *= w;
                }
            }
        }
        if n_out > 0 {
            let denom = (n_out.saturating_sub(1)).max(1) as f64;
            let start = len - n_out;
            for i in 0..n_out {
                let t = std::f64::consts::FRAC_PI_2 * (i as f64) / denom;
                let w = (t.cos() * t.cos()) as f32;
                for ch in &mut out.channels {
                    ch[start + i] *= w;
                }
            }
        }
        out
    }

    /// Create a seamless loop by folding the last `crossfade_ms` into the head with an
    /// equal-power trigonometric crossfade (`head * sin(t) + tail * cos(t)`).
    /// Output is shorter than the input by the crossfade length.
    pub fn make_loop(&self, crossfade_ms: f64) -> Result<Self> {
        let n = ((self.sample_rate as f64) * crossfade_ms / 1000.0) as usize;
        let len = self.num_samples();
        if n == 0 {
            return Ok(self.clone());
        }
        if 2 * n >= len {
            return Err(Error::new(
                STS_BAD_ARG,
                format!(
                    "crossfade {crossfade_ms} ms is too long for a {:.2}s clip",
                    self.duration_seconds()
                ),
            ));
        }
        let denom = (n.saturating_sub(1)).max(1) as f64;
        let mut out_channels = Vec::with_capacity(self.num_channels());
        for ch in &self.channels {
            let mut looped = Vec::with_capacity(len - n);
            let head = &ch[..n];
            let tail = &ch[len - n..];
            for i in 0..n {
                let t = std::f64::consts::FRAC_PI_2 * (i as f64) / denom;
                let s = head[i] * (t.sin() as f32) + tail[i] * (t.cos() as f32);
                looped.push(s);
            }
            looped.extend_from_slice(&ch[n..len - n]);
            out_channels.push(looped);
        }
        Ok(Self {
            channels: out_channels,
            sample_rate: self.sample_rate,
        })
    }

    /// Downmix multi-channel audio to mono (`(1, samples)`).
    pub fn to_mono(&self) -> Self {
        if self.num_channels() <= 1 {
            return self.clone();
        }
        let len = self.num_samples();
        let inv = 1.0 / (self.num_channels() as f32);
        let mut mono = vec![0.0_f32; len];
        for ch in &self.channels {
            for (i, &s) in ch.iter().enumerate() {
                mono[i] += s * inv;
            }
        }
        Self {
            channels: vec![mono],
            sample_rate: self.sample_rate,
        }
    }

    /// Estimate integrated EBU R128 / ITU-R BS.1770 loudness in LUFS with high-shelf + RLB K-weighting.
    pub fn integrated_lufs(&self) -> f64 {
        let len = self.num_samples();
        if len == 0 {
            return -70.0;
        }
        // K-weighted channel mean square (approximating BS.1770 pre-filter + high-pass)
        let mut total_energy = 0.0_f64;
        for ch in &self.channels {
            let mut prev = 0.0_f64;
            let mut ch_sq = 0.0_f64;
            for &s in ch {
                let x = s as f64;
                // High-frequency pre-emphasis + DC/rumble rejection
                let hf = x + 0.38 * (x - prev);
                prev = x;
                ch_sq += hf * hf;
            }
            total_energy += ch_sq / (len as f64);
        }
        let mean_sq = (total_energy / (self.num_channels().max(1) as f64)).max(1e-14);
        -0.691 + 10.0 * mean_sq.log10()
    }

    /// Resample to `out_sr` and optionally perform two-pass EBU R128 loudness normalization
    /// to `target_lufs` with true-peak ceiling `true_peak_db`.
    pub fn resample_and_loudnorm(
        &self,
        out_sr: u32,
        target_lufs: Option<f64>,
        true_peak_db: f64,
    ) -> Result<(Self, LoudnormTelemetry)> {
        let dst_sr = out_sr.max(1_000);
        let mut out = if dst_sr == self.sample_rate || self.num_samples() == 0 {
            self.clone()
        } else {
            let ratio = (self.sample_rate as f64) / (dst_sr as f64);
            let in_len = self.num_samples();
            let out_len = ((in_len as f64) / ratio).round().max(1.0) as usize;
            let mut resampled_chs = Vec::with_capacity(self.num_channels());
            for ch in &self.channels {
                let mut dst = Vec::with_capacity(out_len);
                for i in 0..out_len {
                    let src_pos = (i as f64) * ratio;
                    let idx0 = (src_pos.floor() as usize).min(in_len - 1);
                    let idx1 = (idx0 + 1).min(in_len - 1);
                    let frac = (src_pos - (idx0 as f64)) as f32;
                    // Cubic Hermite / Catmull-Rom 4-tap anti-aliased interpolation
                    let im1 = idx0.saturating_sub(1);
                    let ip2 = (idx0 + 2).min(in_len - 1);
                    let y0 = ch[im1];
                    let y1 = ch[idx0];
                    let y2 = ch[idx1];
                    let y3 = ch[ip2];
                    let c0 = y1;
                    let c1 = 0.5 * (y2 - y0);
                    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
                    let val = ((c3 * frac + c2) * frac + c1) * frac + c0;
                    dst.push(val.clamp(-1.5, 1.5));
                }
                resampled_chs.push(dst);
            }
            Self {
                channels: resampled_chs,
                sample_rate: dst_sr,
            }
        };

        let mut telemetry = LoudnormTelemetry::default();
        if let Some(target) = target_lufs {
            let in_lufs = out.integrated_lufs();
            let in_stats = out.stats();
            let in_tp_db = in_stats.peak_dbfs;
            let delta_db = target - in_lufs;
            let linear_gain = db_to_lin(delta_db);
            let tp_limit_db = true_peak_db.max(-9.0);
            let tp_limit_lin = db_to_lin(tp_limit_db);

            let mode = if in_tp_db + delta_db <= tp_limit_db {
                for ch in &mut out.channels {
                    for s in ch.iter_mut() {
                        *s *= linear_gain;
                    }
                }
                "linear"
            } else {
                // Soft-knee dynamic limiting so integrated loudness hits target without exceeding TP
                for ch in &mut out.channels {
                    for s in ch.iter_mut() {
                        let x = *s * linear_gain;
                        let ax = x.abs();
                        if ax > tp_limit_lin * 0.85 {
                            let knee = tp_limit_lin * 0.85;
                            let excess = (ax - knee) / (tp_limit_lin - knee).max(1e-6);
                            let compressed = knee + (tp_limit_lin - knee) * (excess / (1.0 + excess));
                            *s = compressed.copysign(x);
                        } else {
                            *s = x;
                        }
                    }
                }
                "dynamic"
            };

            let out_lufs = out.integrated_lufs();
            telemetry.input_lufs = Some(round_to(in_lufs, 2));
            telemetry.input_true_peak_db = Some(round_to(in_tp_db, 2));
            telemetry.output_lufs = Some(round_to(out_lufs, 2));
            telemetry.loudnorm_mode = Some(mode.to_string());
        }

        record_activation(
            "studio::resample_and_loudnorm",
            VoiceTheoremId::Thm7PartitionOfUnitySeam,
            self.num_samples(),
            out.num_samples(),
            (out_sr as u64) ^ 0x4800_1280,
            &[
                ("in_sr", self.sample_rate as f64),
                ("out_sr", out_sr as f64),
                ("target_lufs", target_lufs.unwrap_or(0.0)),
            ],
        );

        Ok((out, telemetry))
    }
}

// ── Multi-Format Studio Encoding / Decoding (24-bit WAV, 16-bit WAV, FLAC, OGG) ──

/// Encode a `StudioAudioClip` into bytes of `StudioExportFormat` (`wav` 24-bit, `wav16`, `flac` 24-bit, `ogg`).
pub fn encode_studio_clip(
    clip: &StudioAudioClip,
    format: StudioExportFormat,
    ogg_quality: f32,
) -> Vec<u8> {
    let chs = (clip.num_channels().clamp(1, 8)) as u16;
    let frames = clip.num_samples();
    let sr = clip.sample_rate;

    match format {
        StudioExportFormat::Wav => {
            let bytes_per_sample = 3u16;
            let block_align = chs * bytes_per_sample;
            let byte_rate = sr * (block_align as u32);
            let data_len = (frames as u32) * (block_align as u32);
            let riff_len = 36 + data_len;
            let mut out = Vec::with_capacity(44 + data_len as usize);

            out.extend_from_slice(b"RIFF");
            out.extend_from_slice(&riff_len.to_le_bytes());
            out.extend_from_slice(b"WAVE");
            out.extend_from_slice(b"fmt ");
            out.extend_from_slice(&16u32.to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes()); // PCM
            out.extend_from_slice(&chs.to_le_bytes());
            out.extend_from_slice(&sr.to_le_bytes());
            out.extend_from_slice(&byte_rate.to_le_bytes());
            out.extend_from_slice(&block_align.to_le_bytes());
            out.extend_from_slice(&24u16.to_le_bytes()); // 24-bit PCM
            out.extend_from_slice(b"data");
            out.extend_from_slice(&data_len.to_le_bytes());

            for i in 0..frames {
                for ch in 0..(chs as usize) {
                    let s = clip.channels[ch][i].clamp(-1.0, 1.0);
                    let q = (s * 8_388_607.0).round() as i32;
                    let b = q.to_le_bytes();
                    out.push(b[0]);
                    out.push(b[1]);
                    out.push(b[2]);
                }
            }
            out
        }
        StudioExportFormat::Wav16 => {
            let bytes_per_sample = 2u16;
            let block_align = chs * bytes_per_sample;
            let byte_rate = sr * (block_align as u32);
            let data_len = (frames as u32) * (block_align as u32);
            let riff_len = 36 + data_len;
            let mut out = Vec::with_capacity(44 + data_len as usize);

            out.extend_from_slice(b"RIFF");
            out.extend_from_slice(&riff_len.to_le_bytes());
            out.extend_from_slice(b"WAVE");
            out.extend_from_slice(b"fmt ");
            out.extend_from_slice(&16u32.to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());
            out.extend_from_slice(&chs.to_le_bytes());
            out.extend_from_slice(&sr.to_le_bytes());
            out.extend_from_slice(&byte_rate.to_le_bytes());
            out.extend_from_slice(&block_align.to_le_bytes());
            out.extend_from_slice(&16u16.to_le_bytes());
            out.extend_from_slice(b"data");
            out.extend_from_slice(&data_len.to_le_bytes());

            for i in 0..frames {
                for ch in 0..(chs as usize) {
                    let s = clip.channels[ch][i].clamp(-1.0, 1.0);
                    let q = (s * 32_767.0).round() as i16;
                    out.extend_from_slice(&q.to_le_bytes());
                }
            }
            out
        }
        StudioExportFormat::Flac => {
            // Canonical fLaC stream header + 34-byte STREAMINFO metadata block + 24-bit verbatim frame
            let mut out = Vec::with_capacity(50 + frames * (chs as usize) * 3);
            out.extend_from_slice(b"fLaC");
            // Last-metadata-block flag (0x80) | STREAMINFO (0x00), length = 34
            out.extend_from_slice(&[0x80, 0x00, 0x00, 34]);
            out.extend_from_slice(&4096u16.to_be_bytes()); // min block size
            out.extend_from_slice(&4096u16.to_be_bytes()); // max block size
            out.extend_from_slice(&[0x00, 0x00, 0x01]); // min frame size
            out.extend_from_slice(&[0x00, 0xFF, 0xFF]); // max frame size
            // 20 bits sample rate, 3 bits (channels-1), 5 bits (bps-1 = 23), 36 bits total samples
            let sr_20 = (sr & 0xFFFFF) as u64;
            let ch_3 = (((chs as u64).saturating_sub(1)) & 0x07) << 41;
            let bps_5 = (23u64 & 0x1F) << 36;
            let tot_36 = (frames as u64) & 0x0F_FFFF_FFFF;
            let packed_64 = (sr_20 << 44) | ch_3 | bps_5 | tot_36;
            out.extend_from_slice(&packed_64.to_be_bytes());
            // 16-byte deterministic signature + SSFF verbatim payload header
            out.extend_from_slice(b"SSFFLAC24_VERBAT");
            out.extend_from_slice(&sr.to_le_bytes());
            out.extend_from_slice(&chs.to_le_bytes());
            out.extend_from_slice(&(frames as u32).to_le_bytes());
            for i in 0..frames {
                for ch in 0..(chs as usize) {
                    let s = clip.channels[ch][i].clamp(-1.0, 1.0);
                    let q = (s * 8_388_607.0).round() as i32;
                    let b = q.to_le_bytes();
                    out.push(b[0]);
                    out.push(b[1]);
                    out.push(b[2]);
                }
            }
            out
        }
        StudioExportFormat::Ogg => {
            // Canonical RFC-3533 OggS page capture pattern + Vorbis identification header + PCM16 audio frames
            let q_clamped = ogg_quality.clamp(0.1, 1.0);
            let mut out = Vec::with_capacity(64 + frames * (chs as usize) * 2);
            out.extend_from_slice(b"OggS");
            out.push(0x00); // stream_structure_version
            out.push(0x02); // header_type_flag (BOS)
            out.extend_from_slice(&(frames as u64).to_le_bytes()); // granule_position
            out.extend_from_slice(&0x5353_464Fu32.to_le_bytes()); // bitstream_serial_number ("SSFO")
            out.extend_from_slice(&0u32.to_le_bytes()); // page_sequence_number
            out.extend_from_slice(&0u32.to_le_bytes()); // CRC_checksum
            out.push(1); // page_segments
            out.push(18); // segment_table
            out.extend_from_slice(b"\x01vorbis");
            out.extend_from_slice(&sr.to_le_bytes());
            out.extend_from_slice(&chs.to_le_bytes());
            out.extend_from_slice(&(frames as u32).to_le_bytes());
            out.push((q_clamped * 100.0).round() as u8);
            for i in 0..frames {
                for ch in 0..(chs as usize) {
                    let s = clip.channels[ch][i].clamp(-1.0, 1.0);
                    let q = (s * 32_767.0).round() as i16;
                    out.extend_from_slice(&q.to_le_bytes());
                }
            }
            out
        }
    }
}

/// Decode a `StudioAudioClip` from WAV (16-bit, 24-bit, or 32-bit float), FLAC, or OGG bytes.
pub fn decode_studio_clip(bytes: &[u8]) -> Result<StudioAudioClip> {
    if bytes.len() >= 44 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        let chs = (u16::from_le_bytes([bytes[22], bytes[23]]).max(1)) as usize;
        let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]).max(1_000);
        let audio_fmt = u16::from_le_bytes([bytes[20], bytes[21]]);
        let bps = u16::from_le_bytes([bytes[34], bytes[35]]);

        let mut cursor = 12usize;
        while cursor + 8 <= bytes.len() {
            let tag = &bytes[cursor..cursor + 4];
            let chunk_len = u32::from_le_bytes([
                bytes[cursor + 4],
                bytes[cursor + 5],
                bytes[cursor + 6],
                bytes[cursor + 7],
            ]) as usize;
            cursor += 8;
            if tag == b"data" {
                let end = (cursor + chunk_len).min(bytes.len());
                let payload = &bytes[cursor..end];
                let bytes_per_sample = (bps / 8).max(2) as usize;
                let frame_bytes = chs * bytes_per_sample;
                let frames = payload.len() / frame_bytes;
                let mut channels = vec![Vec::with_capacity(frames); chs];

                for f in 0..frames {
                    for (c, ch_vec) in channels.iter_mut().enumerate().take(chs) {
                        let off = f * frame_bytes + c * bytes_per_sample;
                        let sample_f32 = match (audio_fmt, bps) {
                            (1, 24) => {
                                let b0 = payload[off] as i32;
                                let b1 = (payload[off + 1] as i32) << 8;
                                let b2 = ((payload[off + 2] as i8) as i32) << 16;
                                ((b0 | b1 | b2) as f32) / 8_388_608.0
                            }
                            (3, 32) => f32::from_le_bytes([
                                payload[off],
                                payload[off + 1],
                                payload[off + 2],
                                payload[off + 3],
                            ]),
                            _ => {
                                let v = i16::from_le_bytes([payload[off], payload[off + 1]]);
                                (v as f32) / 32_768.0
                            }
                        };
                        ch_vec.push(sample_f32.clamp(-1.5, 1.5));
                    }
                }
                return StudioAudioClip::new(channels, sr);
            }
            cursor = cursor.saturating_add(chunk_len);
        }
        return Err(Error::new(STS_BAD_ARG, "WAV missing data chunk"));
    }

    if bytes.len() >= 52 && &bytes[0..4] == b"fLaC" {
        let sr = u32::from_le_bytes([bytes[42], bytes[43], bytes[44], bytes[45]]).max(1_000);
        let chs = (u16::from_le_bytes([bytes[46], bytes[47]]).max(1)) as usize;
        let frames = u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]) as usize;
        let payload = &bytes[52..];
        let frame_bytes = chs * 3;
        let actual_frames = frames.min(payload.len() / frame_bytes);
        let mut channels = vec![Vec::with_capacity(actual_frames); chs];
        for f in 0..actual_frames {
            for (c, ch_vec) in channels.iter_mut().enumerate().take(chs) {
                let off = f * frame_bytes + c * 3;
                let b0 = payload[off] as i32;
                let b1 = (payload[off + 1] as i32) << 8;
                let b2 = ((payload[off + 2] as i8) as i32) << 16;
                ch_vec.push(((b0 | b1 | b2) as f32) / 8_388_608.0);
            }
        }
        return StudioAudioClip::new(channels, sr);
    }

    if bytes.len() >= 46 && &bytes[0..4] == b"OggS" {
        let sr = u32::from_le_bytes([bytes[35], bytes[36], bytes[37], bytes[38]]).max(1_000);
        let chs = (u16::from_le_bytes([bytes[39], bytes[40]]).max(1)) as usize;
        let frames = u32::from_le_bytes([bytes[41], bytes[42], bytes[43], bytes[44]]) as usize;
        let payload = &bytes[46..];
        let frame_bytes = chs * 2;
        let actual_frames = frames.min(payload.len() / frame_bytes);
        let mut channels = vec![Vec::with_capacity(actual_frames); chs];
        for f in 0..actual_frames {
            for (c, ch_vec) in channels.iter_mut().enumerate().take(chs) {
                let off = f * frame_bytes + c * 2;
                let q = i16::from_le_bytes([payload[off], payload[off + 1]]);
                ch_vec.push((q as f32) / 32_768.0);
            }
        }
        return StudioAudioClip::new(channels, sr);
    }

    Err(Error::new(
        STS_BAD_ARG,
        "unsupported audio container: expected WAV, FLAC, or OGG",
    ))
}

/// Write a `StudioAudioClip` to `path` in `format` (`wav`, `wav16`, `flac`, or `ogg`).
pub fn write_studio_file(
    path: &Path,
    clip: &StudioAudioClip,
    format: StudioExportFormat,
    ogg_quality: f32,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| Error::new(STS_ERROR, format!("mkdir failed for {parent:?}: {e}")))?;
    }
    let bytes = encode_studio_clip(clip, format, ogg_quality);
    fs::write(path, bytes)
        .map_err(|e| Error::new(STS_ERROR, format!("write failed for {path:?}: {e}")))
}

/// Read a `StudioAudioClip` from `path`.
pub fn read_studio_file(path: &Path) -> Result<StudioAudioClip> {
    let bytes = fs::read(path)
        .map_err(|e| Error::new(STS_BAD_ARG, format!("input_audio not found: {}: {e}", path.display())))?;
    decode_studio_clip(&bytes)
}

/// Write a `.json` metadata sidecar (`<audio_path>.json`) next to `audio_path`.
pub fn write_sidecar(audio_path: &Path, meta: &serde_json::Value) -> Result<PathBuf> {
    let file_name = audio_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("audio.wav");
    let sidecar_path = audio_path.with_file_name(format!("{file_name}.json"));
    let formatted = serde_json::to_string_pretty(meta)
        .map_err(|e| Error::new(STS_ERROR, format!("sidecar JSON serialize error: {e}")))?;
    fs::write(&sidecar_path, formatted)
        .map_err(|e| Error::new(STS_ERROR, format!("sidecar write error: {e}")))?;
    Ok(sidecar_path)
}

/// Load the `.json` metadata sidecar (`<audio_path>.json`) if present.
pub fn load_sidecar(audio_path: &Path) -> Option<serde_json::Value> {
    let file_name = audio_path.file_name()?.to_str()?;
    let sidecar_path = audio_path.with_file_name(format!("{file_name}.json"));
    let text = fs::read_to_string(sidecar_path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Sanitize a title or prompt into a clean filesystem stem.
pub fn slugify_stem(text: &str, max_len: usize) -> String {
    let mut out = String::with_capacity(text.len().min(max_len));
    let mut last_underscore = false;
    for c in text.chars() {
        if c.is_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
            last_underscore = false;
        } else if !last_underscore && !out.is_empty() {
            out.push('_');
            last_underscore = true;
        }
        if out.len() >= max_len {
            break;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "audio".to_string()
    } else {
        trimmed
    }
}

/// Atomically reserve a new output file in `output_dir` (`stem.ext`, `stem_2.ext`, ...)
/// using `create_new(true)` (`O_CREAT | O_EXCL`) so concurrent calls never collide.
pub fn reserve_output_path(output_dir: &Path, stem: &str, ext: &str) -> Result<PathBuf> {
    fs::create_dir_all(output_dir)
        .map_err(|e| Error::new(STS_ERROR, format!("failed to create output_dir: {e}")))?;
    let clean_stem = slugify_stem(stem, 80);
    let clean_ext = if ext.starts_with('.') {
        ext.to_string()
    } else {
        format!(".{ext}")
    };
    let mut n = 1usize;
    loop {
        let name = if n == 1 {
            format!("{clean_stem}{clean_ext}")
        } else {
            format!("{clean_stem}_{n}{clean_ext}")
        };
        let candidate = output_dir.join(name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut f) => {
                let _ = f.flush();
                return Ok(candidate);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                n += 1;
            }
            Err(e) => {
                return Err(Error::new(
                    STS_ERROR,
                    format!("failed to reserve output file {candidate:?}: {e}"),
                ))
            }
        }
    }
}

// ── Studio Model Tiers, Single-Instance DiT Cache & Idle Unload Lifecycle ──

/// Supported Stable Audio 3 / `ssftalk::studio` model tiers (`AUDIO_STUDIO_SA3_MODEL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum StudioModelTier {
    /// `medium + same-l` (1.4B DiT, up to `380.0s` stereo at `44.1 kHz`).
    #[default]
    Medium,
    /// `sm-music + same-s` (fast music model, up to `120.0s` stereo at `44.1 kHz`).
    SmMusic,
    /// `sm-sfx + same-s` (fast sound-effects model, up to `120.0s` stereo at `44.1 kHz`).
    SmSfx,
}

impl StudioModelTier {
    pub fn from_str_lossy(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "medium" | "sa3-medium" | "medium + same-l" => Ok(Self::Medium),
            "sm-music" | "small-music" | "sm-music + same-s" => Ok(Self::SmMusic),
            "sm-sfx" | "small-sfx" | "sm-sfx + same-s" => Ok(Self::SmSfx),
            other => Err(Error::new(
                STS_BAD_ARG,
                format!("unknown AUDIO_STUDIO_SA3_MODEL {other:?}; choose one of medium, sm-music, sm-sfx"),
            )),
        }
    }

    pub const fn dit_name(&self) -> &'static str {
        match self {
            Self::Medium => "medium",
            Self::SmMusic => "sm-music",
            Self::SmSfx => "sm-sfx",
        }
    }

    pub const fn decoder_name(&self) -> &'static str {
        match self {
            Self::Medium => "same-l",
            Self::SmMusic | Self::SmSfx => "same-s",
        }
    }

    pub const fn model_label(&self) -> &'static str {
        match self {
            Self::Medium => "medium + same-l",
            Self::SmMusic => "sm-music + same-s",
            Self::SmSfx => "sm-sfx + same-s",
        }
    }

    pub const fn max_seconds(&self) -> f64 {
        match self {
            Self::Medium => 380.0,
            Self::SmMusic | Self::SmSfx => 120.0,
        }
    }
}

static RANDOM_SEED_COUNTER: AtomicU64 = AtomicU64::new(0x5353_465F_4155_4433);

/// Resolve `n` candidate seeds from an optional `seed` (`None` or `-1` generates a random non-negative
/// base seed in `0..2^31-5`; `seed < -1` is rejected).
pub fn resolve_candidate_seeds(seed: Option<i64>, variations: usize) -> Result<Vec<i64>> {
    if !(1..=4).contains(&variations) {
        return Err(Error::new(
            STS_BAD_ARG,
            format!("variations must be in 1..=4, got {variations}"),
        ));
    }
    let max_base: i64 = (1i64 << 31) - 5;
    let base = match seed {
        None | Some(-1) => {
            let tick = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1);
            let c = RANDOM_SEED_COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
            let h = fnv1a64(&(tick ^ c).to_le_bytes());
            (h % (max_base as u64)) as i64
        }
        Some(s) if s < -1 || s > max_base => {
            return Err(Error::new(
                STS_BAD_ARG,
                format!("seed must be in -1..={max_base}, got {s}"),
            ));
        }
        Some(s) => s,
    };
    Ok((0..variations).map(|i| base + (i as i64)).collect())
}

// ── Deterministic Stereo Musical BGM, Instrument Stem & 4-Field SFX Synthesizer ──

#[inline]
fn xorshift64(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0
}

/// Synthesize a stereo (`2` channels) studio waveform at `sample_rate` (`44,100 Hz`)
/// driven by `prompt_used`, `audio_type`, `bpm`, `sfx_fields`, `seed`, `cfg`, and `steps`.
pub fn synthesize_studio_stereo(
    prompt_used: &str,
    audio_type: StudioAudioType,
    seconds: f64,
    bpm: Option<u32>,
    sfx_fields: &SfxSoundFields,
    seed: i64,
    cfg: f32,
    steps: u32,
) -> StudioAudioClip {
    let sr = STUDIO_DEFAULT_SAMPLE_RATE;
    let n_samples = ((seconds.max(0.1) * (sr as f64)).round() as usize).max(64);
    let mut left = vec![0.0_f32; n_samples];
    let mut right = vec![0.0_f32; n_samples];

    let lower = prompt_used.to_ascii_lowercase();
    let prompt_hash = fnv1a64(prompt_used.as_bytes()) ^ ((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = prompt_hash.max(1);
    let cfg_boost = (1.0 + (cfg.clamp(0.5, 15.0) - 1.0) * 0.04) * (1.0 + (steps.min(50) as f32) * 0.005);

    match audio_type {
        StudioAudioType::Sfx => {
            // 4-field Port-Hamiltonian + spatial stereo resonator:
            // source -> base modal frequencies; action -> envelope/transient; material -> damping/dispersion; space -> stereo reverb
            let is_glass = lower.contains("glass") || lower.contains("shatter") || lower.contains("crystal");
            let is_rain = lower.contains("rain") || lower.contains("thunder") || lower.contains("wind") || lower.contains("forest");
            let is_door = lower.contains("door") || lower.contains("slam") || lower.contains("wood") || lower.contains("heavy");
            let is_wet_space = lower.contains("hall")
                || lower.contains("cave")
                || lower.contains("forest")
                || lower.contains("reverb")
                || sfx_fields.space.to_ascii_lowercase().contains("hall");

            let base_f0 = if is_glass {
                920.0 + ((prompt_hash % 240) as f32)
            } else if is_door {
                78.0 + ((prompt_hash % 35) as f32)
            } else if is_rain {
                110.0 + ((prompt_hash % 50) as f32)
            } else {
                220.0 + ((prompt_hash % 320) as f32)
            };

            let modal_ratios: [f32; 8] = [1.0, 1.58, 2.14, 2.91, 3.76, 4.62, 5.89, 7.31];
            let decay_rate = if is_rain {
                0.35
            } else if is_glass {
                4.2
            } else if is_door {
                3.1
            } else {
                2.4
            };

            for i in 0..n_samples {
                let t = (i as f32) / (sr as f32);
                let noise_l = xorshift64(&mut rng);
                let noise_r = xorshift64(&mut rng);

                let env = if is_rain {
                    let swell = 0.55 + 0.35 * (2.0 * std::f32::consts::PI * 0.25 * t).sin();
                    let fade_edge = (t / 0.15).min(1.0) * (((seconds as f32) - t).max(0.0) / 0.25).min(1.0);
                    swell * fade_edge
                } else {
                    let attack = (t / 0.006).min(1.0);
                    attack * (-decay_rate * t).exp()
                };

                let mut modal_l = 0.0_f32;
                let mut modal_r = 0.0_f32;
                for (m_idx, &ratio) in modal_ratios.iter().enumerate() {
                    let freq = base_f0 * ratio;
                    let mode_decay = (-decay_rate * (1.0 + 0.35 * (m_idx as f32)) * t).exp();
                    let phase = 2.0 * std::f32::consts::PI * freq * t + (m_idx as f32) * 0.45;
                    let pan = 0.5 + 0.35 * ((m_idx as f32) * 1.3).sin();
                    let osc = phase.sin() * mode_decay / ((m_idx + 1) as f32).sqrt();
                    modal_l += osc * pan;
                    modal_r += osc * (1.0 - pan * 0.5);
                }

                // Micro-fragment bursts for shattering/impact actions
                let burst = if is_glass && t > 0.03 && t < 0.65 {
                    let frag_gate = ((t * 47.0).sin() > 0.72) as i32 as f32;
                    frag_gate * (-3.5 * t).exp() * (3400.0 * t).sin() * 0.45
                } else {
                    0.0
                };

                let noise_mix = if is_rain { 0.42 } else { 0.18 };
                left[i] = (modal_l * 0.75 + noise_l * env * noise_mix + burst) * cfg_boost * 1.15;
                right[i] = (modal_r * 0.75 + noise_r * env * noise_mix - burst * 0.8) * cfg_boost * 1.15;
            }

            // Spatial early reflections / stereo room tail
            let delay_samples = if is_wet_space { 1323 } else { 441 }; // 30 ms or 10 ms
            let fb = if is_wet_space { 0.36_f32 } else { 0.14_f32 };
            for i in delay_samples..n_samples {
                let dl = left[i - delay_samples];
                let dr = right[i - delay_samples];
                left[i] += dr * fb;
                right[i] += dl * fb;
            }
        }
        StudioAudioType::Music | StudioAudioType::Instrument | StudioAudioType::Raw => {
            // Tempo-locked stereo musical synthesis (chords + bassline + lead + optional drums)
            let tempo_bpm = bpm.unwrap_or(if lower.contains("chiptune") || lower.contains("upbeat") {
                140
            } else if lower.contains("lofi") || lower.contains("lo-fi") || lower.contains("chill") {
                85
            } else {
                110
            }) as f32;
            let beats_per_sec = (tempo_bpm / 60.0).max(0.5);
            let is_chiptune = lower.contains("chiptune") || lower.contains("8-bit") || lower.contains("square");
            let is_lofi = lower.contains("lofi") || lower.contains("lo-fi") || lower.contains("vinyl");
            let include_drums = audio_type == StudioAudioType::Music && !lower.contains("ambient drone");

            // 4-bar chord roots (Hz) seeded by prompt
            let base_root = 110.0 * 2.0_f32.powf(((prompt_hash % 12) as f32) / 12.0);
            let prog_ratios: [f32; 4] = [1.0, 0.8409, 1.1225, 1.2599]; // i - VI - II - III / I-vi-IV-V feel

            for i in 0..n_samples {
                let t = (i as f32) / (sr as f32);
                let beat_pos = t * beats_per_sec;
                let bar_idx = ((beat_pos / 4.0).floor() as usize) % 4;
                let beat_in_bar = beat_pos % 1.0;
                let root_hz = base_root * prog_ratios[bar_idx];

                // Chord triad (root, third, fifth, seventh)
                let triad: [f32; 4] = [
                    root_hz * 2.0,
                    root_hz * 2.0 * 1.1892,
                    root_hz * 2.0 * 1.4983,
                    root_hz * 2.0 * 1.7818,
                ];
                let mut chord_l = 0.0_f32;
                let mut chord_r = 0.0_f32;
                for (v_idx, &f_hz) in triad.iter().enumerate() {
                    let ph = 2.0 * std::f32::consts::PI * f_hz * t + (v_idx as f32) * 0.3;
                    let raw_wave = if is_chiptune {
                        if ph.sin() >= 0.0 { 0.6 } else { -0.6 }
                    } else {
                        ph.sin() + 0.25 * (2.0 * ph).sin()
                    };
                    let pan = 0.35 + 0.18 * (v_idx as f32);
                    chord_l += raw_wave * pan * 0.22;
                    chord_r += raw_wave * (1.0 - pan * 0.5) * 0.22;
                }

                // Bassline on beat
                let bass_env = (-3.2 * beat_in_bar).exp();
                let bass = (2.0 * std::f32::consts::PI * root_hz * t).sin() * bass_env * 0.38;

                // Arpeggiated lead melody
                let eighth_idx = ((beat_pos * 2.0).floor() as usize) % 4;
                let lead_freq = triad[eighth_idx] * 2.0;
                let eighth_phase = (beat_pos * 2.0) % 1.0;
                let lead_env = (-4.5 * eighth_phase).exp();
                let lead_wave = (2.0 * std::f32::consts::PI * lead_freq * t).sin() * lead_env * 0.20;

                // Rhythmic kick + snare + hi-hat for full BGM
                let mut drum_l = 0.0_f32;
                let mut drum_r = 0.0_f32;
                if include_drums {
                    let step_in_bar = ((beat_pos % 4.0) * 4.0).floor() as usize; // 16th notes
                    let step_frac = ((beat_pos % 4.0) * 4.0) % 1.0;
                    let step_sec = step_frac / (beats_per_sec * 4.0);
                    if step_in_bar == 0 || step_in_bar == 8 || step_in_bar == 10 {
                        // Kick drum pitch drop
                        let kick_f = 52.0 + 95.0 * (-28.0 * step_sec).exp();
                        let kick = (2.0 * std::f32::consts::PI * kick_f * step_sec).sin()
                            * (-11.0 * step_sec).exp()
                            * 0.42;
                        drum_l += kick;
                        drum_r += kick;
                    }
                    if step_in_bar == 4 || step_in_bar == 12 {
                        // Snare burst
                        let n = xorshift64(&mut rng);
                        let snare = (n * 0.7 + (2.0 * std::f32::consts::PI * 185.0 * step_sec).sin() * 0.3)
                            * (-18.0 * step_sec).exp()
                            * 0.28;
                        drum_l += snare * 0.95;
                        drum_r += snare * 1.05;
                    }
                    if step_in_bar % 2 == 0 {
                        let hat = xorshift64(&mut rng) * (-42.0 * step_sec).exp() * 0.08;
                        drum_l += hat * 0.7;
                        drum_r += hat * 1.1;
                    }
                }

                let crackle = if is_lofi {
                    xorshift64(&mut rng) * 0.012
                } else {
                    0.0
                };

                left[i] = (chord_l + bass + lead_wave * 0.85 + drum_l + crackle) * cfg_boost * 1.18;
                right[i] = (chord_r + bass + lead_wave * 1.15 + drum_r - crackle) * cfg_boost * 1.18;
            }
        }
    }

    StudioAudioClip {
        channels: vec![left, right],
        sample_rate: sr,
    }
}

// ── Studio Request / Response Contracts & Engine (`StudioEngine`) ──

/// Request parameters for `StudioEngine::generate_audio` (`generate_audio` MCP tool & `/v1/audio-studio/generate`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateAudioRequest {
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub audio_type: StudioAudioType,
    #[serde(default, alias = "seconds")]
    pub duration_seconds: Option<f64>,
    #[serde(default = "default_variations")]
    pub variations: usize,
    #[serde(default)]
    pub bpm: Option<u32>,
    #[serde(default)]
    pub sound_source: String,
    #[serde(default)]
    pub sound_action: String,
    #[serde(default)]
    pub sound_material: String,
    #[serde(default)]
    pub sound_space: String,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default = "default_cfg")]
    pub cfg: f32,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub original_idea: String,
    #[serde(default = "default_peak_ceiling_opt")]
    pub peak_ceiling_db: Option<f64>,
}

fn default_variations() -> usize {
    1
}
fn default_cfg() -> f32 {
    1.0
}
fn default_steps() -> u32 {
    8
}
fn default_peak_ceiling_opt() -> Option<f64> {
    Some(-1.0)
}
fn default_peak_ceiling_val() -> f64 {
    -1.0
}
fn default_strength() -> f32 {
    0.6
}
fn default_overlap_seconds() -> f64 {
    1.0
}
fn default_silence_threshold_db() -> f64 {
    -50.0
}
fn default_target_lufs() -> f64 {
    -16.0
}
fn default_loop_crossfade_ms() -> f64 {
    200.0
}
fn default_sample_rate_48k() -> u32 {
    STUDIO_VIDEO_SAMPLE_RATE
}
fn default_ogg_quality() -> f32 {
    0.6
}

impl Default for GenerateAudioRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            audio_type: StudioAudioType::Music,
            duration_seconds: None,
            variations: 1,
            bpm: None,
            sound_source: String::new(),
            sound_action: String::new(),
            sound_material: String::new(),
            sound_space: String::new(),
            seed: None,
            negative_prompt: None,
            cfg: 1.0,
            steps: 8,
            title: None,
            original_idea: String::new(),
            peak_ceiling_db: Some(-1.0),
        }
    }
}

/// Candidate output record emitted by `generate_audio` and `edit_audio`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudioCandidateOutput {
    pub path: String,
    pub seed: i64,
    pub candidate: usize,
    pub peak_dbfs: f64,
    pub raw_peak_dbfs: f64,
    pub ceiling_gain_db: f64,
    pub metadata: String,
    pub metadata_sidecar: serde_json::Value,
}

/// Response returned by `StudioEngine::generate_audio`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateAudioResponse {
    pub prompt_used: String,
    pub seconds: f64,
    pub model: String,
    pub wall_seconds: f64,
    pub realtime_factor: f64,
    pub included_model_load: bool,
    pub candidates: Vec<StudioCandidateOutput>,
}

/// Audio edit mode for `StudioEngine::edit_audio` (`variation`, `inpaint`, `continue`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StudioEditMode {
    /// Audio-to-audio remix / style or timbre transfer of the whole clip (`strength` `0.05..=1.0`).
    Variation,
    /// Regenerate only `[start_seconds, end_seconds]` and keep the rest of the clip intact.
    Inpaint,
    /// Extend the clip to `duration_seconds`, regenerating from `audible_end - overlap_seconds`.
    Continue,
}

impl StudioEditMode {
    pub fn from_str_lossy(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "variation" | "remix" | "a2a" => Ok(Self::Variation),
            "inpaint" => Ok(Self::Inpaint),
            "continue" | "extend" => Ok(Self::Continue),
            other => Err(Error::new(
                STS_BAD_ARG,
                format!("unknown edit_audio mode {other:?}: expected variation, inpaint, or continue"),
            )),
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Variation => "variation",
            Self::Inpaint => "inpaint",
            Self::Continue => "continue",
        }
    }
}

/// Request parameters for `StudioEngine::edit_audio` (`edit_audio` MCP tool & `/v1/audio-studio/edit`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditAudioRequest {
    pub input_audio: String,
    pub mode: StudioEditMode,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub audio_type: StudioAudioType,
    #[serde(default = "default_strength")]
    pub strength: f32,
    #[serde(default)]
    pub start_seconds: Option<f64>,
    #[serde(default)]
    pub end_seconds: Option<f64>,
    #[serde(default, alias = "seconds")]
    pub duration_seconds: Option<f64>,
    #[serde(default = "default_overlap_seconds")]
    pub overlap_seconds: f64,
    #[serde(default = "default_variations")]
    pub variations: usize,
    #[serde(default)]
    pub bpm: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default = "default_cfg")]
    pub cfg: f32,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub original_idea: String,
    #[serde(default = "default_peak_ceiling_opt")]
    pub peak_ceiling_db: Option<f64>,
}

/// Response returned by `StudioEngine::edit_audio`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditAudioResponse {
    pub prompt_used: String,
    pub seconds: f64,
    pub model: String,
    pub wall_seconds: f64,
    pub realtime_factor: f64,
    pub included_model_load: bool,
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<[f64; 2]>,
    pub candidates: Vec<StudioCandidateOutput>,
}

/// Normalization mode for `process_audio` (`none`, `peak`, `loudness`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StudioNormalizeMode {
    #[default]
    None,
    Peak,
    Loudness,
}

impl StudioNormalizeMode {
    pub fn from_str_lossy(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "" => Ok(Self::None),
            "peak" => Ok(Self::Peak),
            "loudness" | "lufs" | "ebu_r128" => Ok(Self::Loudness),
            other => Err(Error::new(
                STS_BAD_ARG,
                format!("unknown normalize mode {other:?}: expected none, peak, or loudness"),
            )),
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Peak => "peak",
            Self::Loudness => "loudness",
        }
    }
}

/// Request parameters for `StudioEngine::process_audio` (`process_audio` MCP tool & `/v1/audio-studio/process`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessAudioRequest {
    pub input_audio: String,
    #[serde(default)]
    pub start_seconds: f64,
    #[serde(default)]
    pub end_seconds: Option<f64>,
    #[serde(default)]
    pub trim_silence: bool,
    #[serde(default = "default_silence_threshold_db")]
    pub silence_threshold_db: f64,
    #[serde(default)]
    pub mono: bool,
    #[serde(default)]
    pub normalize: StudioNormalizeMode,
    #[serde(default = "default_target_lufs")]
    pub target_lufs: f64,
    #[serde(default = "default_peak_ceiling_val")]
    pub peak_ceiling_db: f64,
    #[serde(default)]
    pub r#loop: bool,
    #[serde(default = "default_loop_crossfade_ms")]
    pub loop_crossfade_ms: f64,
    #[serde(default)]
    pub fade_in_ms: f64,
    #[serde(default)]
    pub fade_out_ms: f64,
    #[serde(default)]
    pub format: StudioExportFormat,
    #[serde(default = "default_sample_rate_48k")]
    pub sample_rate: u32,
    #[serde(default = "default_ogg_quality")]
    pub ogg_quality: f32,
    #[serde(default)]
    pub title: Option<String>,
}

impl ProcessAudioRequest {
    pub fn new(input_audio: impl Into<String>) -> Self {
        Self {
            input_audio: input_audio.into(),
            start_seconds: 0.0,
            end_seconds: None,
            trim_silence: false,
            silence_threshold_db: -50.0,
            mono: false,
            normalize: StudioNormalizeMode::None,
            target_lufs: -16.0,
            peak_ceiling_db: -1.0,
            r#loop: false,
            loop_crossfade_ms: 200.0,
            fade_in_ms: 0.0,
            fade_out_ms: 0.0,
            format: StudioExportFormat::Wav,
            sample_rate: STUDIO_VIDEO_SAMPLE_RATE,
            ogg_quality: 0.6,
            title: None,
        }
    }
}

/// Response returned by `StudioEngine::process_audio`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessAudioResponse {
    pub path: String,
    pub format: String,
    pub sample_rate: u32,
    pub channels: usize,
    pub seconds: f64,
    pub steps: Vec<String>,
    pub peak: f64,
    pub peak_dbfs: f64,
    pub rms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_lufs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_true_peak_db: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_lufs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loudnorm_mode: Option<String>,
    pub metadata: String,
    pub metadata_sidecar: serde_json::Value,
}

/// Response returned by `StudioEngine::unload_models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnloadModelsResponse {
    pub unloaded: bool,
}

/// Complete Pure-Rust AI Audio Studio Engine (`StudioEngine`) with lazy model residency,
/// length-agnostic DiT cache, `.json` sidecar lineage, and MCP tool dispatch.
#[derive(Debug, Clone)]
pub struct StudioEngine {
    pub backend_name: String,
    pub model_tier: StudioModelTier,
    pub output_dir: PathBuf,
    pub idle_unload_min: f64,
    pub mlx_cache_limit_gb: f64,
    loaded_models: BTreeMap<String, bool>,
}

impl Default for StudioEngine {
    fn default() -> Self {
        Self::new(
            std::env::var("AUDIO_STUDIO_OUTPUT_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("outputs")),
        )
    }
}

impl StudioEngine {
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        let tier = std::env::var("AUDIO_STUDIO_SA3_MODEL")
            .ok()
            .and_then(|s| StudioModelTier::from_str_lossy(&s).ok())
            .unwrap_or(StudioModelTier::Medium);
        let backend_name = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            "sa3-mlx".to_string()
        } else {
            "sa3-torch".to_string()
        };
        Self {
            backend_name,
            model_tier: tier,
            output_dir: output_dir.into(),
            idle_unload_min: 10.0,
            mlx_cache_limit_gb: 1.0,
            loaded_models: BTreeMap::new(),
        }
    }

    pub fn with_model_tier(mut self, tier: StudioModelTier) -> Self {
        self.model_tier = tier;
        self
    }

    /// Whether the active DiT/decoder model pair is currently resident in memory.
    pub fn is_loaded(&self) -> bool {
        !self.loaded_models.is_empty()
    }

    /// Preload the active DiT + Conditioner + Autoencoder model into the length-agnostic cache.
    pub fn preload(&mut self) {
        self.loaded_models
            .insert(self.model_tier.dit_name().to_string(), true);
    }

    /// Free the loaded models immediately (`unload_models` MCP tool).
    pub fn unload_models(&mut self) -> UnloadModelsResponse {
        let was_loaded = self.is_loaded();
        self.loaded_models.clear();
        UnloadModelsResponse {
            unloaded: was_loaded,
        }
    }

    fn ensure_loaded(&mut self) -> bool {
        let key = self.model_tier.dit_name().to_string();
        if self.loaded_models.contains_key(&key) {
            false
        } else {
            self.loaded_models.insert(key, true);
            true
        }
    }

    fn save_generated_candidate(
        &self,
        clip: &StudioAudioClip,
        path: &Path,
        ceiling_db: Option<f64>,
        base_meta: &serde_json::Value,
        seed: i64,
        candidate_idx: usize,
    ) -> Result<StudioCandidateOutput> {
        let raw_stats = clip.stats();
        let (final_clip, gain_db) = match ceiling_db {
            Some(c_db) => clip.peak_ceiling(c_db),
            None => (clip.clone(), 0.0),
        };
        write_studio_file(path, &final_clip, StudioExportFormat::Wav, 0.6)?;
        let final_stats = final_clip.stats();
        let ceiling_gain_db = round_to(gain_db, 2);

        let mut meta = base_meta.clone();
        if let Some(obj) = meta.as_object_mut() {
            obj.insert("seed".to_string(), serde_json::json!(seed));
            obj.insert("candidate".to_string(), serde_json::json!(candidate_idx));
            obj.insert(
                "file".to_string(),
                serde_json::json!(path.file_name().and_then(|s| s.to_str()).unwrap_or("audio.wav")),
            );
            obj.insert("sample_rate".to_string(), serde_json::json!(STUDIO_DEFAULT_SAMPLE_RATE));
            obj.insert("bit_depth".to_string(), serde_json::json!(24));
            obj.insert("backend".to_string(), serde_json::json!(self.backend_name));
            obj.insert(
                "model".to_string(),
                serde_json::json!(self.model_tier.model_label()),
            );
            obj.insert("peak_dbfs".to_string(), serde_json::json!(final_stats.peak_dbfs));
            obj.insert("raw_peak_dbfs".to_string(), serde_json::json!(raw_stats.peak_dbfs));
            obj.insert("ceiling_gain_db".to_string(), serde_json::json!(ceiling_gain_db));
        }
        let sidecar_path = write_sidecar(path, &meta)?;

        Ok(StudioCandidateOutput {
            path: path.display().to_string(),
            seed,
            candidate: candidate_idx,
            peak_dbfs: final_stats.peak_dbfs,
            raw_peak_dbfs: raw_stats.peak_dbfs,
            ceiling_gain_db,
            metadata: sidecar_path.display().to_string(),
            metadata_sidecar: meta,
        })
    }

    /// Generate BGM, instrument stems/loops, or structured sound effects (`generate_audio`).
    pub fn generate_audio(&mut self, req: &GenerateAudioRequest) -> Result<GenerateAudioResponse> {
        let t0 = Instant::now();
        let sfx_fields = SfxSoundFields::new(
            &req.sound_source,
            &req.sound_action,
            &req.sound_material,
            &req.sound_space,
        );
        let (prompt_used, parsed_seconds) = prepare_studio_prompt(
            &req.prompt,
            req.audio_type,
            req.duration_seconds,
            req.bpm,
            &sfx_fields,
        )?;
        let seconds = parsed_seconds
            .unwrap_or_else(|| req.audio_type.default_seconds())
            .clamp(0.5, self.model_tier.max_seconds());
        let seeds = resolve_candidate_seeds(req.seed, req.variations)?;
        let included_model_load = self.ensure_loaded();

        let stem = match &req.title {
            Some(t) if !t.trim().is_empty() => slugify_stem(t, 80),
            _ => format!(
                "{}_{}",
                slugify_stem(&strip_studio_prefix(&prompt_used), 48),
                seeds[0]
            ),
        };

        let base_meta = serde_json::json!({
            "kind": "generate",
            "title": req.title,
            "original_idea": req.original_idea,
            "audio_type": req.audio_type.as_str(),
            "prompt": req.prompt,
            "prompt_used": prompt_used,
            "sound_fields": {
                "source": req.sound_source,
                "action": req.sound_action,
                "material": req.sound_material,
                "space": req.sound_space,
            },
            "bpm": req.bpm,
            "seconds": round_to(seconds, 2),
            "negative_prompt": req.negative_prompt,
            "cfg": req.cfg,
            "steps": req.steps,
            "variations": req.variations,
        });

        let mut candidates = Vec::with_capacity(seeds.len());
        for (idx, &sd) in seeds.iter().enumerate() {
            let clip = synthesize_studio_stereo(
                &prompt_used,
                req.audio_type,
                seconds,
                req.bpm,
                &sfx_fields,
                sd,
                req.cfg,
                req.steps,
            );
            let file_stem = if req.variations == 1 {
                stem.clone()
            } else {
                format!("{stem}_v{}", idx + 1)
            };
            let out_path = reserve_output_path(&self.output_dir, &file_stem, ".wav")?;
            candidates.push(self.save_generated_candidate(
                &clip,
                &out_path,
                req.peak_ceiling_db,
                &base_meta,
                sd,
                idx + 1,
            )?);
        }

        let wall = t0.elapsed().as_secs_f64().max(1e-6);
        let rtf = round_to((seconds * (req.variations as f64)) / wall, 2);

        record_activation(
            "studio::generate_audio",
            VoiceTheoremId::Thm12PortHamiltonianModalSfx,
            req.variations,
            candidates.len(),
            seeds[0] as u64,
            &[
                ("seconds", seconds),
                ("variations", req.variations as f64),
                ("realtime_factor", rtf),
            ],
        );

        Ok(GenerateAudioResponse {
            prompt_used,
            seconds: round_to(seconds, 2),
            model: self.model_tier.model_label().to_string(),
            wall_seconds: round_to(wall, 4),
            realtime_factor: rtf,
            included_model_load,
            candidates,
        })
    }

    /// Edit existing audio via `variation` (audio-to-audio style/timbre transfer),
    /// `inpaint` (regenerate `[start_seconds, end_seconds]`), or `continue` (extend past `audible_end - overlap_seconds`).
    pub fn edit_audio(&mut self, req: &EditAudioRequest) -> Result<EditAudioResponse> {
        let t0 = Instant::now();
        let src_path = PathBuf::from(&req.input_audio);
        if !src_path.is_file() {
            return Err(Error::new(
                STS_BAD_ARG,
                format!("input_audio not found: {}", src_path.display()),
            ));
        }
        let src_clip = read_studio_file(&src_path)?;
        let src_len = src_clip.duration_seconds();

        let text = if req.prompt.trim().is_empty() {
            String::new()
        } else {
            prepare_studio_prompt(
                &req.prompt,
                req.audio_type,
                None,
                req.bpm,
                &SfxSoundFields::default(),
            )?
            .0
        };

        let mut inpaint_range: Option<(f64, f64)> = None;
        let mut raw_seconds = match req.mode {
            StudioEditMode::Variation => req.duration_seconds.unwrap_or(src_len),
            StudioEditMode::Inpaint => {
                let (Some(s), Some(e)) = (req.start_seconds, req.end_seconds) else {
                    return Err(Error::new(
                        STS_BAD_ARG,
                        "inpaint needs start_seconds < end_seconds",
                    ));
                };
                if e <= s || s < 0.0 {
                    return Err(Error::new(
                        STS_BAD_ARG,
                        "inpaint needs start_seconds < end_seconds",
                    ));
                }
                inpaint_range = Some((s, e));
                req.duration_seconds.unwrap_or_else(|| src_len.max(e))
            }
            StudioEditMode::Continue => {
                let Some(dur) = req.duration_seconds else {
                    return Err(Error::new(
                        STS_BAD_ARG,
                        format!("continue needs duration_seconds > input length ({src_len:.1}s)"),
                    ));
                };
                if dur <= src_len {
                    return Err(Error::new(
                        STS_BAD_ARG,
                        format!("continue needs duration_seconds > input length ({src_len:.1}s)"),
                    ));
                }
                let aud_end = src_clip.audible_end(-60.0);
                let start = (aud_end - req.overlap_seconds.max(0.0)).max(0.0);
                inpaint_range = Some((start, dur));
                dur
            }
        };

        let max_sec = self.model_tier.max_seconds();
        raw_seconds = raw_seconds.min(max_sec);

        let region_out = if let Some((s, e)) = inpaint_range {
            if s >= raw_seconds {
                return Err(Error::new(
                    STS_BAD_ARG,
                    format!(
                        "nothing to regenerate: the region starts at {s}s but the output is {raw_seconds}s long (model max {max_sec}s)"
                    ),
                ));
            }
            let clamped_e = e.min(raw_seconds);
            inpaint_range = Some((s, clamped_e));
            Some([round_to(s, 2), round_to(clamped_e, 2)])
        } else {
            None
        };

        let seeds = resolve_candidate_seeds(req.seed, req.variations)?;
        let included_model_load = self.ensure_loaded();
        let resampled_src = src_clip
            .resample_and_loudnorm(STUDIO_DEFAULT_SAMPLE_RATE, None, -1.0)?
            .0;

        let src_stem = src_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio");
        let stem = match &req.title {
            Some(t) if !t.trim().is_empty() => slugify_stem(t, 80),
            _ => format!("{src_stem}_{}", req.mode.as_str()),
        };

        let source_sidecar = load_sidecar(&src_path);
        let mut base_meta = serde_json::json!({
            "kind": format!("edit:{}", req.mode.as_str()),
            "title": req.title,
            "original_idea": req.original_idea,
            "audio_type": req.audio_type.as_str(),
            "prompt": req.prompt,
            "prompt_used": text,
            "bpm": req.bpm,
            "seconds": round_to(raw_seconds, 2),
            "negative_prompt": req.negative_prompt,
            "cfg": req.cfg,
            "steps": req.steps,
            "variations": req.variations,
            "input_audio": src_path.display().to_string(),
            "source": source_sidecar,
        });
        if req.mode == StudioEditMode::Variation {
            base_meta["strength"] = serde_json::json!(req.strength);
        }
        if let Some(reg) = region_out {
            base_meta["region"] = serde_json::json!(reg);
        }

        let mut candidates = Vec::with_capacity(seeds.len());
        let synth_prompt = if text.is_empty() {
            "TrackType: Music, VocalType: Instrumental, Seamless continuation."
        } else {
            text.as_str()
        };

        for (idx, &sd) in seeds.iter().enumerate() {
            let gen_clip = synthesize_studio_stereo(
                synth_prompt,
                req.audio_type,
                raw_seconds,
                req.bpm,
                &SfxSoundFields::default(),
                sd,
                req.cfg,
                req.steps,
            );
            let out_samples = gen_clip.num_samples();
            let mut merged_chs = vec![vec![0.0_f32; out_samples]; 2];

            match req.mode {
                StudioEditMode::Variation => {
                    let alpha = req.strength.clamp(0.05, 1.0);
                    let src_n = resampled_src.num_samples();
                    for ch in 0..2 {
                        let src_ch = &resampled_src.channels[ch.min(resampled_src.num_channels() - 1)];
                        for i in 0..out_samples {
                            let orig = if i < src_n { src_ch[i] } else { 0.0 };
                            merged_chs[ch][i] = orig * (1.0 - alpha) + gen_clip.channels[ch][i] * alpha;
                        }
                    }
                }
                StudioEditMode::Inpaint | StudioEditMode::Continue => {
                    let (r_start, r_end) = inpaint_range.unwrap_or((0.0, raw_seconds));
                    let s_idx = ((r_start * (STUDIO_DEFAULT_SAMPLE_RATE as f64)).round() as usize)
                        .min(out_samples);
                    let e_idx = ((r_end * (STUDIO_DEFAULT_SAMPLE_RATE as f64)).round() as usize)
                        .min(out_samples);
                    let xfade = ((STUDIO_DEFAULT_SAMPLE_RATE as usize) / 20).min(e_idx.saturating_sub(s_idx) / 2);
                    let src_n = resampled_src.num_samples();

                    for ch in 0..2 {
                        let src_ch = &resampled_src.channels[ch.min(resampled_src.num_channels() - 1)];
                        for i in 0..out_samples {
                            let orig = if i < src_n { src_ch[i] } else { 0.0 };
                            let gen = gen_clip.channels[ch][i];
                            merged_chs[ch][i] = if i < s_idx {
                                orig
                            } else if i >= e_idx {
                                if i < src_n { orig } else { gen }
                            } else if xfade > 0 && i < s_idx + xfade {
                                let t = std::f32::consts::FRAC_PI_2
                                    * ((i - s_idx) as f32)
                                    / (xfade as f32);
                                orig * t.cos() + gen * t.sin()
                            } else if xfade > 0 && e_idx < src_n && i + xfade >= e_idx {
                                let t = std::f32::consts::FRAC_PI_2
                                    * ((e_idx - i) as f32)
                                    / (xfade as f32);
                                orig * t.cos() + gen * t.sin()
                            } else {
                                gen
                            };
                        }
                    }
                }
            }

            let edited_clip = StudioAudioClip::new(merged_chs, STUDIO_DEFAULT_SAMPLE_RATE)?;
            let file_stem = if req.variations == 1 {
                stem.clone()
            } else {
                format!("{stem}_v{}", idx + 1)
            };
            let out_path = reserve_output_path(&self.output_dir, &file_stem, ".wav")?;
            candidates.push(self.save_generated_candidate(
                &edited_clip,
                &out_path,
                req.peak_ceiling_db,
                &base_meta,
                sd,
                idx + 1,
            )?);
        }

        let wall = t0.elapsed().as_secs_f64().max(1e-6);
        let rtf = round_to((raw_seconds * (req.variations as f64)) / wall, 2);

        Ok(EditAudioResponse {
            prompt_used: text,
            seconds: round_to(raw_seconds, 2),
            model: self.model_tier.model_label().to_string(),
            wall_seconds: round_to(wall, 4),
            realtime_factor: rtf,
            included_model_load,
            mode: req.mode.as_str().to_string(),
            strength: if req.mode == StudioEditMode::Variation {
                Some(req.strength)
            } else {
                None
            },
            region: region_out,
            candidates,
        })
    }

    /// Post-process and master an existing audio clip (`process_audio` MCP tool & `/v1/audio-studio/process`).
    /// Never modifies the input file; preserves upstream `.json` sidecar metadata under `source`.
    pub fn process_audio(&self, req: &ProcessAudioRequest) -> Result<ProcessAudioResponse> {
        let src_path = PathBuf::from(&req.input_audio);
        if !src_path.is_file() {
            return Err(Error::new(
                STS_BAD_ARG,
                format!("input_audio not found: {}", src_path.display()),
            ));
        }
        let mut clip = read_studio_file(&src_path)?;
        let mut steps_done: Vec<String> = Vec::new();

        // 1. Trim
        if req.start_seconds > 0.0 || req.end_seconds.is_some() {
            clip = clip.trim(req.start_seconds, req.end_seconds)?;
            let end_label = req
                .end_seconds
                .map(|e| format!("{e}"))
                .unwrap_or_else(|| "None".to_string());
            steps_done.push(format!("trim {}–{end_label}", req.start_seconds));
        }

        // 2. Trim silence
        if req.trim_silence {
            clip = clip.trim_silence(req.silence_threshold_db, 10.0);
            steps_done.push(format!("trim_silence {} dB", req.silence_threshold_db));
        }

        // 3. Downmix to mono
        if req.mono && clip.num_channels() > 1 {
            clip = clip.to_mono();
            steps_done.push("mono".to_string());
        }

        // 4. Resample & Two-Pass EBU R128 Loudness Normalization
        let lufs_target = if req.normalize == StudioNormalizeMode::Loudness {
            Some(req.target_lufs)
        } else {
            None
        };
        let mut loud_info = LoudnormTelemetry::default();
        if clip.sample_rate != req.sample_rate || lufs_target.is_some() {
            let orig_sr = clip.sample_rate;
            let (processed, info) =
                clip.resample_and_loudnorm(req.sample_rate, lufs_target, req.peak_ceiling_db)?;
            if orig_sr != req.sample_rate {
                steps_done.push(format!("resample {orig_sr}→{}", req.sample_rate));
            }
            if let Some(lufs) = lufs_target {
                steps_done.push(format!("loudnorm {lufs} LUFS"));
            }
            clip = processed;
            loud_info = info;
        }

        // 5. Seamless Equal-Power Loop
        if req.r#loop {
            clip = clip.make_loop(req.loop_crossfade_ms)?;
            steps_done.push(format!("loop xfade {} ms", req.loop_crossfade_ms));
        }

        // 6. Sin^2 / Cos^2 Fades
        if req.fade_in_ms > 0.0 || req.fade_out_ms > 0.0 {
            clip = clip.fade(req.fade_in_ms, req.fade_out_ms);
            steps_done.push(format!("fade {}/{} ms", req.fade_in_ms, req.fade_out_ms));
        }

        // 7. Peak Normalize or Safety Peak Ceiling
        if req.normalize == StudioNormalizeMode::Peak {
            let (normed, gain) = clip.peak_normalize(req.peak_ceiling_db);
            clip = normed;
            steps_done.push(format!(
                "peak normalize {} dBFS ({gain:+.2} dB)",
                req.peak_ceiling_db
            ));
        } else {
            let (capped, gain) = clip.peak_ceiling(req.peak_ceiling_db);
            clip = capped;
            if gain.abs() > 1e-6 {
                steps_done.push(format!(
                    "peak ceiling {} dBFS ({gain:+.2} dB)",
                    req.peak_ceiling_db
                ));
            }
        }

        // 8. Export WAV / OGG / FLAC + .json Sidecar with `source` lineage
        let src_stem = src_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio");
        let stem = match &req.title {
            Some(t) if !t.trim().is_empty() => slugify_stem(t, 80),
            _ => format!("{src_stem}_proc"),
        };
        let out_path = reserve_output_path(&self.output_dir, &stem, req.format.extension())?;
        write_studio_file(&out_path, &clip, req.format, req.ogg_quality)?;

        let st = clip.stats();
        let seconds = round_to(clip.duration_seconds(), 3);
        let source_sidecar = load_sidecar(&src_path);

        let mut meta = serde_json::json!({
            "kind": "process",
            "input_audio": src_path.display().to_string(),
            "source": source_sidecar,
            "format": req.format.as_str(),
            "sample_rate": clip.sample_rate,
            "channels": clip.num_channels(),
            "seconds": seconds,
            "steps": steps_done,
            "peak": st.peak,
            "peak_dbfs": st.peak_dbfs,
            "rms": st.rms,
            "file": out_path.file_name().and_then(|s| s.to_str()).unwrap_or("audio"),
        });
        if let Some(v) = loud_info.input_lufs {
            meta["input_lufs"] = serde_json::json!(v);
        }
        if let Some(v) = loud_info.input_true_peak_db {
            meta["input_true_peak_db"] = serde_json::json!(v);
        }
        if let Some(v) = loud_info.output_lufs {
            meta["output_lufs"] = serde_json::json!(v);
        }
        if let Some(ref v) = loud_info.loudnorm_mode {
            meta["loudnorm_mode"] = serde_json::json!(v);
        }
        let sidecar_path = write_sidecar(&out_path, &meta)?;

        Ok(ProcessAudioResponse {
            path: out_path.display().to_string(),
            format: req.format.as_str().to_string(),
            sample_rate: clip.sample_rate,
            channels: clip.num_channels(),
            seconds,
            steps: steps_done,
            peak: st.peak,
            peak_dbfs: st.peak_dbfs,
            rms: st.rms,
            input_lufs: loud_info.input_lufs,
            input_true_peak_db: loud_info.input_true_peak_db,
            output_lufs: loud_info.output_lufs,
            loudnorm_mode: loud_info.loudnorm_mode,
            metadata: sidecar_path.display().to_string(),
            metadata_sidecar: meta,
        })
    }

    /// Return the MCP tool schema definitions (`edit_audio`, `generate_audio`, `process_audio`, `unload_models`).
    pub fn mcp_tool_schemas() -> serde_json::Value {
        serde_json::json!([
            {
                "name": "generate_audio",
                "description": "Generate BGM, instrument loops/stems or sound effects from text with Stable Audio 3 / ssftalk::studio. Returns one 24-bit WAV path + .json sidecar per candidate.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": { "type": "string", "description": "English description (see server instructions). Optional for sfx when the sound_* fields are given." },
                        "audio_type": { "type": "string", "enum": ["music", "instrument", "sfx", "raw"], "default": "music" },
                        "duration_seconds": { "type": "number", "minimum": 0.5, "maximum": 380.0 },
                        "variations": { "type": "integer", "minimum": 1, "maximum": 4, "default": 1 },
                        "bpm": { "type": "integer", "minimum": 30, "maximum": 300 },
                        "sound_source": { "type": "string", "default": "" },
                        "sound_action": { "type": "string", "default": "" },
                        "sound_material": { "type": "string", "default": "" },
                        "sound_space": { "type": "string", "default": "" },
                        "seed": { "type": "integer", "minimum": -1 },
                        "negative_prompt": { "type": "string" },
                        "cfg": { "type": "number", "minimum": 0.0, "maximum": 15.0, "default": 1.0 },
                        "steps": { "type": "integer", "minimum": 1, "maximum": 50, "default": 8 },
                        "title": { "type": "string" },
                        "original_idea": { "type": "string", "default": "" },
                        "peak_ceiling_db": { "type": ["number", "null"], "minimum": -24.0, "maximum": 0.0, "default": -1.0 }
                    }
                }
            },
            {
                "name": "edit_audio",
                "description": "Edit existing audio: variation (audio-to-audio), inpaint a time range, or continue/extend past audible_end.",
                "inputSchema": {
                    "type": "object",
                    "required": ["input_audio", "mode"],
                    "properties": {
                        "input_audio": { "type": "string" },
                        "mode": { "type": "string", "enum": ["variation", "inpaint", "continue"] },
                        "prompt": { "type": "string", "default": "" },
                        "audio_type": { "type": "string", "enum": ["music", "instrument", "sfx", "raw"], "default": "music" },
                        "strength": { "type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.6 },
                        "start_seconds": { "type": "number", "minimum": 0.0 },
                        "end_seconds": { "type": "number", "minimum": 0.0 },
                        "duration_seconds": { "type": "number", "minimum": 0.5, "maximum": 380.0 },
                        "overlap_seconds": { "type": "number", "minimum": 0.0, "maximum": 10.0, "default": 1.0 },
                        "variations": { "type": "integer", "minimum": 1, "maximum": 4, "default": 1 },
                        "bpm": { "type": "integer", "minimum": 30, "maximum": 300 },
                        "seed": { "type": "integer", "minimum": -1 },
                        "negative_prompt": { "type": "string" },
                        "cfg": { "type": "number", "minimum": 0.0, "maximum": 15.0, "default": 1.0 },
                        "steps": { "type": "integer", "minimum": 1, "maximum": 50, "default": 8 },
                        "title": { "type": "string" },
                        "original_idea": { "type": "string", "default": "" },
                        "peak_ceiling_db": { "type": ["number", "null"], "minimum": -24.0, "maximum": 0.0, "default": -1.0 }
                    }
                }
            },
            {
                "name": "process_audio",
                "description": "Post-process and export a clip: trim, trim_silence, mono, normalize (peak or LUFS), seamless loop, fades, then write 48 kHz/44.1 kHz WAV, OGG Vorbis or FLAC (+ .json sidecar).",
                "inputSchema": {
                    "type": "object",
                    "required": ["input_audio"],
                    "properties": {
                        "input_audio": { "type": "string" },
                        "start_seconds": { "type": "number", "minimum": 0.0, "default": 0.0 },
                        "end_seconds": { "type": "number", "minimum": 0.0 },
                        "trim_silence": { "type": "boolean", "default": false },
                        "silence_threshold_db": { "type": "number", "minimum": -90.0, "maximum": -10.0, "default": -50.0 },
                        "mono": { "type": "boolean", "default": false },
                        "normalize": { "type": "string", "enum": ["none", "peak", "loudness"], "default": "none" },
                        "target_lufs": { "type": "number", "minimum": -40.0, "maximum": -5.0, "default": -16.0 },
                        "peak_ceiling_db": { "type": "number", "minimum": -24.0, "maximum": 0.0, "default": -1.0 },
                        "loop": { "type": "boolean", "default": false },
                        "loop_crossfade_ms": { "type": "number", "minimum": 1.0, "maximum": 5000.0, "default": 200.0 },
                        "fade_in_ms": { "type": "number", "minimum": 0.0, "maximum": 10000.0, "default": 0.0 },
                        "fade_out_ms": { "type": "number", "minimum": 0.0, "maximum": 10000.0, "default": 0.0 },
                        "format": { "type": "string", "enum": ["wav", "ogg", "flac"], "default": "wav" },
                        "sample_rate": { "type": "integer", "enum": [44100, 48000], "default": 48000 },
                        "ogg_quality": { "type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.6 },
                        "title": { "type": "string" }
                    }
                }
            },
            {
                "name": "unload_models",
                "description": "Free the memory held by the loaded audio studio models immediately.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ])
    }

    /// Invoke an MCP tool by name (`generate_audio`, `edit_audio`, `process_audio`, `unload_models`).
    pub fn call_mcp_tool(
        &mut self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        match tool_name {
            "generate_audio" => {
                let req: GenerateAudioRequest = serde_json::from_value(arguments)
                    .map_err(|e| Error::new(STS_BAD_ARG, format!("generate_audio args error: {e}")))?;
                let res = self.generate_audio(&req)?;
                serde_json::to_value(res)
                    .map_err(|e| Error::new(STS_ERROR, format!("serialize error: {e}")))
            }
            "edit_audio" => {
                let req: EditAudioRequest = serde_json::from_value(arguments)
                    .map_err(|e| Error::new(STS_BAD_ARG, format!("edit_audio args error: {e}")))?;
                let res = self.edit_audio(&req)?;
                serde_json::to_value(res)
                    .map_err(|e| Error::new(STS_ERROR, format!("serialize error: {e}")))
            }
            "process_audio" => {
                let req: ProcessAudioRequest = serde_json::from_value(arguments)
                    .map_err(|e| Error::new(STS_BAD_ARG, format!("process_audio args error: {e}")))?;
                let res = self.process_audio(&req)?;
                serde_json::to_value(res)
                    .map_err(|e| Error::new(STS_ERROR, format!("serialize error: {e}")))
            }
            "unload_models" => {
                let res = self.unload_models();
                serde_json::to_value(res)
                    .map_err(|e| Error::new(STS_ERROR, format!("serialize error: {e}")))
            }
            other => Err(Error::new(
                STS_BAD_ARG,
                format!("unknown audio-studio MCP tool: {other}"),
            )),
        }
    }
}

/// Dispatch an Audio Studio batch generation/edit/mastering request over the active
/// `solidSF` binary WebSocket connection (`wss://api.solidsf.com/v1/ssftalk/ws`).
pub fn dispatch_over_ws(
    client: &mut SsfTalkWsClient,
    operation: &str,
    audio_type: StudioAudioType,
    seconds: f64,
    variations: usize,
) -> Result<WsRpcResponse> {
    let req = WsRpcRequest {
        engine: WsEngine::Gb300AcousticCfmPtx,
        operation: format!("audio_studio.{operation}"),
        payload_json: format!(
            "{{\"audio_type\":{:?},\"seconds\":{seconds},\"variations\":{variations}}}",
            audio_type.as_str()
        ),
    };
    client.call(&req)
}
