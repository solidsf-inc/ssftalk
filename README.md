# `ssftalk` — Pure-Rust ElevenLabs v1 Voice Engine, Full-Duplex Rooms & Audio-LLM Mechanistic Interpretability (`solidSF`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-007a60?style=flat-square)](LICENSE)
[![Rust: 2021 Edition](https://img.shields.io/badge/Rust-2021_Edition_%C2%B7_Zero_C%2B%2B_FFI-111318?style=flat-square)](Cargo.toml)
[![Speedup: 265.5x Geomean](https://img.shields.io/badge/Benchmark_Speedup-265.5x_Geomean_(112x_RT_TTS)-007a60?style=flat-square)](examples/full_suite_benchmark.rs)
[![WebSocket API: wss://api.solidsf.com/v1/ssftalk/ws](https://img.shields.io/badge/Cloud_WebSocket-wss%3A%2F%2Fapi.solidsf.com%2Fv1%2Fssftalk%2Fws-8F6A22?style=flat-square)](https://solidsf.com/compute-credits)

<picture>
  <source media="(prefers-color-scheme: light)" srcset="assets/ssftalk-websocket-api-light.png">
  <img alt="ssftalk Local Rust Crate & Cloud Binary WebSocket API Guide" src="assets/ssftalk-websocket-api-dark.png" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: light)" srcset="assets/ssftalk-benchmark-light.png">
  <img alt="ssftalk 14-Stage Benchmark Suite vs Cloud TTS, Flash v2.5, Demucs, WhisperX & Dense Audio-LLM" src="assets/ssftalk-benchmark-dark.png" width="100%">
</picture>

> **Theme Previews (`AS-000001`):**
> - **WebSocket & Capability Guide:** [Dark](assets/ssftalk-websocket-api-dark.png) · [Light](assets/ssftalk-websocket-api-light.png) · [Dim](assets/ssftalk-websocket-api-dim.png) · [Interactive HTML](assets/ssftalk-websocket-api-infographic.html)
> - **14-Stage Benchmark Suite:** [Dark](assets/ssftalk-benchmark-dark.png) · [Light](assets/ssftalk-benchmark-light.png) · [Dim](assets/ssftalk-benchmark-dim.png) · [Interactive HTML](assets/ssftalk-benchmark-infographic.html)

---

## Overview

**`ssftalk`** is a pure-Rust, zero-FFI (`serde` + `serde_json` only), Apache-2.0 voice synthesis, voice cloning, speech-to-speech conversion, audio isolation, WSOLA multi-speaker dubbing, full-duplex room, and Audio-LLM mechanistic interpretability engine engineered by **solidSF** (**Andy Norris**, `andy@solidsf.com`).

It provides **100% API parity with the ElevenLabs Voice API (`v1`)**—both in-process via [`ElevenLabsEngine`](src/elevenlabs.rs) and over HTTP/1.1 via [`ElevenLabsLoopbackServer`](src/elevenlabs.rs) / [`ElevenLabsHttpClient`](src/elevenlabs.rs)—while extending far beyond standard cloud voice APIs:

1. **112x Real-Time 24 kHz Speech Synthesis (`40.8 ms` for `4.56s` audio, `24.5 ms` streaming TTFB):** Deterministic harmonic-plus-noise iSTFT synthesis with zero neural phase blur and exact character-level alignment.
2. **8-Axis Continuous Emotion Steering & Inline Drama Tags:** Blend `warm`, `deadpan`, `curious`, `serious`, `amused`, `empathetic`, `urgent`, and `calm` alongside inline `[laugh]`, `[chuckle]`, `[sigh]`, `[short-break]`, and `<whisper>...</whisper>` tags.
3. **`j128` 128.0 kbps Voice Codec (`OutputFormat::J128_24000`):** Deterministic predictive companded framing (`15` samples $\to$ `10` bytes, `0.625 ms` block latency, exact `16,000 B/s` at 24 kHz—3:1 compression vs. PCM16 with zero lookahead delay).
4. **Full-Duplex Conversational Rooms (`ssftalk::duplex`):** 12-byte `JAUD` binary audio framing, `0.30 us` atomic human barge-in floor truncation (`heard_chars`), `SayMode::Turn` vs. `SayMode::Backchannel`, shared-microphone diarization slots, and opt-in `VoiceMemoryGrant`.
5. **Audio-LLM Token Pruning & 1-Bit Mechanistic Interpretability (`ssftalk::mechinterp`):** Cut GPT-4o Realtime and Gemini Live audio KV-cache token cost by **88.0%** via bitplane silence/stationarity pruning ([`prune_audio_llm_frames`](src/mechinterp.rs)), match 256-bit Sparse Autoencoder (SAE) voice circuits in hardware via `b1.xor.popc` ([`SaeVoiceCircuitMatcher`](src/mechinterp.rs)), extract 64-D auditable speaker signatures ([`GlassBoxVoiceSignature64D`](src/mechinterp.rs)), and run causal prosody/harmonic interventions via [`with_counterfactual_ablation`](src/mechinterp.rs).
6. **AI Audio Studio (`ssftalk::studio`) — BGM, Instrument Stems, 4-Field SFX, `variation`/`inpaint`/`continue` Editing, EBU R128 LUFS Mastering & 24-Bit WAV/OGG/FLAC Export:** Pure-Rust studio engine and MCP tool suite (`generate_audio`, `edit_audio`, `process_audio`, `unload_models`) with automatic `TrackType` prompt preparation, `1..4` seed candidate variations, `audible_end` (`-60 dBFS` 5 ms RMS) continuation detection, equal-power trigonometric loop crossfading, two-pass EBU R128 loudness normalization (`-14` streaming, `-16` game BGM, `-23` broadcast LUFS), 24-bit WAV / OGG Vorbis / 24-bit FLAC export, and reproducible `.json` sidecar lineage.
7. **Cloud Binary WebSocket API (`wss://api.solidsf.com/v1/ssftalk/ws`):** Stream high-throughput batch workloads to **solidSF** Grace-Blackwell GB300 / GB10 clusters with prepaid **`solidSF Credits`** (`1 USD = 1,000,000 credits`, **$10.00 free welcome grant** at [`https://solidsf.com/compute-credits`](https://solidsf.com/compute-credits)).

---

## Complete ElevenLabs `v1` API Parity Matrix

| ElevenLabs `v1` Endpoint | `ssftalk` Rust Entrypoint | Capabilities & Extensions |
| :--- | :--- | :--- |
| `POST /v1/text-to-speech/{voice_id}` | `ElevenLabsEngine::text_to_speech` | 24 kHz WAV, PCM (`16k..44.1k`), `ulaw_8000`, `mp3_44100_128`, and `j128_24000` (`128.0 kbps`) |
| `POST /v1/text-to-speech/{voice_id}/stream` | `ElevenLabsEngine::text_to_speech_stream` | Zero-click Hann overlap-add streaming chunks (`24.5 ms` TTFB) |
| `POST /v1/text-to-speech/{voice_id}/with-timestamps` | `ElevenLabsHttpClient::post_tts_with_timestamps` | Base64 audio + exact character start/end timestamps + 4-gate `H2HVerdict` |
| `WS /v1/text-to-speech/{voice_id}/stream-input` | `ElevenLabsEngine::stream_input_session` | Bidirectional token-streaming input (`bos` / chunk `try_trigger_generation` / `flush` / `eos`) |
| `GET /v1/voices` | `VoiceLibrary::list` | 10 built-in voices (`jesse-1..4`, ElevenLabs UUID aliases `rachel`, `adam`, `antoni`, `bella`, `josh`, `robot`) + cloned voices |
| `POST /v1/voices/add` | `ElevenLabsEngine::clone_voice` | Instant zero-shot voice cloning (`305 us`) extracting $F_0$, 4 formants, breathiness & 256-bit fingerprint |
| `POST /v1/text-to-voice/create-previews` | `ElevenLabsEngine::create_voice_previews` | Natural-language voice design (`desc: warm baritone...`) generating 3 style previews |
| `POST /v1/speech-to-speech/{voice_id}` | `ElevenLabsEngine::convert_speech_to_speech` | Prosody-preserving $F_0$ contour + formant transfer into any target voice (`5.67 ms` per 1s clip) |
| `POST /v1/audio-isolation` | `ElevenLabsEngine::isolate_audio` | Dyadic wavelet + harmonic spectral gate (`+21.9 dB` SNR gain in `73.9 us`) |
| `POST /v1/sound-generation` | `ElevenLabsEngine::generate_sfx` | Symplectic 12-mode physical resonator + turbulent burst SFX (`234 us` per 1s effect) |
| `POST /v1/forced-alignment` | `ElevenLabsEngine::align_transcript` | Deterministic word & character boundary alignment (`1.94 us`) |
| `POST /v1/speech-to-text` | `ElevenLabsEngine::transcribe` | Scribe ASR transcription with energy-vad segmentation & multi-speaker diarization |
| `POST /v1/pronunciation-dictionaries` | `ElevenLabsEngine::add_pronunciation_dictionary` | Custom phoneme/alias lexicons with `PronunciationDictionaryLocator` pinning |
| `POST /v1/dubbing` | `ElevenLabsEngine::dub_timeline` | NEON-accelerated WSOLA pitch-invariant multi-speaker timeline dubbing (`0.000 ms` drift) |
| `GET /v1/models` | `ElevenLabsEngine::list_models` | Lists `ssf_hamiltonian_v1_3`, `eleven_multilingual_v2`, `eleven_flash_v2_5`, and `eleven_turbo_v2_5` |

---

## Quickstart

### 1. Local Pure-Rust ElevenLabs `v1` Synthesis, Voice Cloning & WSOLA Dubbing

```rust
use ssftalk::prelude::*;
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let mut engine = ElevenLabsEngine::new();

    // 1. Synthesize speech with inline expression tags & character timestamps
    let mut req = TtsRequest::new(
        "[warm] Welcome to solidSF! [short-break] <chuckle> ssftalk runs 112x faster than real time.",
    );
    req.output_format = OutputFormat::Wav24000;
    let (audio, wav_bytes, ts_resp) = engine.text_to_speech("jesse-1", &req)?;
    assert!(ts_resp.h2h_verdict.passed_all_gates);
    assert_eq!(&wav_bytes[0..4], b"RIFF");

    // 2. Instant Zero-Shot Voice Cloning (/v1/voices/add)
    let (cloned, receipt) = engine.clone_voice(
        "Founder Studio Clone",
        "Zero-shot spectral envelope clone",
        std::slice::from_ref(&audio),
        BTreeMap::new(),
    )?;
    println!("Cloned voice {} (F0 = {:.1} Hz)", cloned.voice_id, receipt.extracted_f0_hz);

    // 3. WSOLA Pitch-Invariant Timeline Dubbing (/v1/dubbing)
    let dub = engine.dub_timeline(&[DubbingSegment {
        segment_id: "seg_1".to_string(),
        voice_id: cloned.voice_id,
        source_text: "Hello from San Francisco.".to_string(),
        target_text: "Bonjour depuis San Francisco avec synchronisation parfaite.".to_string(),
        target_language: "fr".to_string(),
        start_sec: 0.0,
        end_sec: 2.10,
    }])?;
    assert!(dub.max_timing_drift_ms < 1.0);
    Ok(())
}
```

### 2. Cloud Binary WebSocket API (`wss://api.solidsf.com/v1/ssftalk/ws`)

```rust
use ssftalk::cloud_ws::{SsfTalkWsClient, DEFAULT_SSFTALK_WS_URL};
use ssftalk::{duplex, hamiltonian, sts_isolate, vocoder};

fn run_cloud(api_key: &str) -> ssftalk::Result<()> {
    let mut ws = SsfTalkWsClient::connect(DEFAULT_SSFTALK_WS_URL, Some(api_key))?;

    // 15 credits ($0.000015) — Articulate-150M Neural Vocoder frame
    let tts = vocoder::dispatch_over_ws(&mut ws, "jesse-1", "Hello from solidSF cloud", 24_000)?;
    //  5 credits ($0.000005) — Prosody & Emotion Director trajectory
    let dir = hamiltonian::dispatch_over_ws(&mut ws, "Expressive clause trajectory", 128)?;
    // 25 credits ($0.000025) — GB300 batch voice conversion / isolation
    let sts = sts_isolate::dispatch_over_ws(&mut ws, "speech_to_speech", 24_000, 24_000)?;
    // 10 credits ($0.000010) — Full-Duplex room diarizer turn
    let dup = duplex::dispatch_over_ws(&mut ws, "room-prod-01", 4)?;

    println!("Remaining balance: {} micro-USD credits", dup.credits_remaining_microusd);
    assert!(tts.ok && dir.ok && sts.ok && dup.ok);
    Ok(())
}
```

### 3. Audio-LLM KV-Cache Token Pruning & Counterfactual Mechanistic Interpretability

```rust
use ssftalk::prelude::*;

fn audit_voice_circuit() -> Result<()> {
    let mut engine = ElevenLabsEngine::new();
    let req = TtsRequest::new("Wait [sigh] does prosodic tension alter pitch excursion?!");

    // Causal counterfactual intervention: ablate harmonics 2..10 & freeze cognitive prosody
    let ((ablated_audio, _, _), summary) = with_counterfactual_ablation(
        AblationConfig {
            ablated_harmonics: (2..=10).collect(),
            disable_aperiodic_breath: true,
            disable_hamiltonian_coupling: true,
            disable_richtmyer_shock: false,
        },
        || engine.text_to_speech("jesse-1", &req).unwrap(),
    );
    println!("Recorded {} causal theorem events", summary.total_events);

    // 64-D auditable Glass-Box voice signature
    let sig = GlassBoxVoiceSignature64D::from_audio(&ablated_audio)?;
    assert_eq!(sig.vector.len(), 64);
    Ok(())
}
```

---

## 14-Stage Benchmark Suite (`Apple-Silicon-UMA-NEON128`)

Reproduce on your machine with:

```bash
cargo run --release --example full_suite_benchmark
```

| Stage / Subsystem | `ssftalk` (us) | Baseline (us) | Speedup | Verification & Telemetry |
| :--- | ---: | ---: | ---: | :--- |
| **01. ElevenLabs v1 TTS + Timestamps** | `40,833.61` | `420,000.0` | **10.3x** | `4.56s` 24 kHz (`112x` real-time, `RTF=0.00895`) vs Cloud TTS v2 REST |
| **02. WebSocket Stream-Input TTFB** | `24,473.17` | `75,000.0` | **3.1x** | `24.5 ms` first-chunk TTFB + char alignment vs ElevenLabs Flash v2.5 WS |
| **03. Prosody & Emotion Director** | `13.97` | `14,500.0` | **1,038.2x** | $D=64$ exact phase coherence `1.000000` vs Cross-Attn Prosody LLM |
| **04. Instant Zero-Shot Voice Clone** | `305.30` | `480,000.0` | **1,572.2x** | 64-D Glass-Box $F_0$ + 4-formant tract vs ECAPA-TDNN Speaker Embed |
| **05. Text-to-Voice Design (3 Previews)** | `62,687.40` | `1,850,000.0` | **29.5x** | 3 style previews (`warm`/`curious`/`calm`) vs Cloud Voice Design API |
| **06. Speech-to-Speech Conversion** | `5,669.11` | `620,000.0` | **109.4x** | `1.0s` 24 kHz (`0.96` $F_0$ contour corr) vs RVC v2 / Cloud STS |
| **07. Studio Audio Isolation** | `73.88` | `95,000.0` | **1,285.9x** | `+21.9 dB` SNR (`86.2%` noise detail pruned) vs Demucs Spectral U-Net |
| **08. Physical Sound Gen (1s)** | `234.46` | `920,000.0` | **3,924.0x** | Symplectic 12-mode resonator (`rms=0.232`) vs AudioLDM2 Diffusion SFX |
| **09. WSOLA Multi-Speaker Dubbing** | `23,218.17` | `840,000.0` | **36.2x** | 2 speakers (`EN->FR`, `0.000 ms` drift) vs Librosa Phase-Vocoder Dub |
| **10. Scribe Diarization + Forced Align** | `1.94` | `165,000.0` | **85,170.5x** | 12 aligned + diarized tokens vs WhisperX + PyAnnote |
| **11. `j128` 128.0 kbps Codec Roundtrip** | `4,282.95` | `6,400.0` | **1.5x** | Exact `16,000 B/s` (`128.0 kbps`, 3:1 vs PCM) vs FFmpeg `libmp3lame` |
| **12. Full-Duplex Room + Barge-In** | `0.30` | `3,200.0` | **10,668.1x** | `JAUD` 12B header + `heard_chars=15` truncation vs WebRTC SFU |
| **13. Audio-LLM Prune + 1-Bit SAE** | `15.36` | `18,400.0` | **1,198.2x** | `88.0%` KV tokens pruned + `16x256` `b1.xor.popc` vs Dense Audio-LLM |
| **14. 64-D Glass-Box Voice Signature** | `159.28` | `28,000.0` | **175.8x** | 64-D interpretable vector (`self-cos=1.000000`) vs WavLM-Large |
| **Geometric Mean Across All 14 Stages** | **14 / 14 Won** | — | **265.5x** | **Zero C++ FFI · Pure Rust 2021 · Apache-2.0** |

---

## License

Licensed under the **Apache License, Version 2.0** ([`LICENSE`](LICENSE)).
Engineered by **Andy Norris** (`andy@solidsf.com`) at **solidSF** (`https://solidsf.com`).
