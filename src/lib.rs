//! `ssftalk`: Pure-Rust, Apache-2.0 Drop-In **ElevenLabs Voice API (`v1`)** Equivalent
//! and Next-Generation Hamiltonian-Fourier Speech, Voice Cloning, Audio Isolation,
//! WSOLA Dubbing, Full-Duplex Conversational Room, and Audio-LLM Mechanistic Interpretability Engine (`solidSF`).
//!
//! Replaces rate-limited cloud-only voice APIs and heavyweight Python/C++ audio stacks with
//! zero-FFI, hardware-accelerated Rust operators (Apple Silicon NEON + 16 KB Metal 3.1 UMA slabs,
//! NVIDIA Grace-Blackwell GB300 / GB10 PTX 8.5+ Tensor Cores) grounded in the 12 theorems of
//! the Hamiltonian-Fourier & Acoustic Calculus and instrumented with first-class Mechanistic
//! Interpretability (`mechinterp`) telemetry and counterfactual prosody/harmonic ablation hooks.

pub mod cloud_ws;
pub mod codec;
pub mod core;
pub mod duplex;
pub mod elevenlabs;
pub mod hamiltonian;
pub mod hardware;
pub mod mechinterp;
pub mod phonetics;
pub mod prelude;
pub mod sts_isolate;
pub mod studio;
pub mod vocoder;
pub mod voices;

pub use crate::core::{AudioBuffer, Error, OutputFormat, Result};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elevenlabs::{
        ElevenLabsEngine, ElevenLabsHttpClient, ElevenLabsLoopbackServer, TtsRequest,
        WsStreamInputMessage,
    };
    use crate::mechinterp::{reset_mind, with_counterfactual_ablation, AblationConfig};
    use crate::phonetics::{PronunciationDictionaryLocator, PronunciationEntry};
    use crate::sts_isolate::DubbingSegment;
    use std::collections::BTreeMap;

    #[test]
    fn hamiltonian_fourier_phase_binding_and_cognitive_trajectory_hold() {
        reset_mind();
        let role = hamiltonian::FourierPhaseVector::from_token("agent_subject");
        let filler = hamiltonian::FourierPhaseVector::from_token("hypersonic_turbine");
        let bound = role.bind_role(&filler);
        let recovered = bound.unbind_role(&role);
        assert!(
            recovered.phase_coherence(&filler) > 0.999999,
            "conjugate unbinding must be exact on unit-modulus torus"
        );

        let traj = hamiltonian::ThinkingTrajectory::from_transcript(
            "The baseline is slow, but however ssftalk executes in microseconds!",
            64,
        );
        assert_eq!(traj.semantic_fg_mask.len(), 64);
        assert!((traj.hodge_closure - 1.0).abs() < 1e-6);
        let max_shock = traj
            .richtmyer_shock
            .iter()
            .fold(0.0_f32, |a, &b| a.max(b));
        assert!(max_shock >= 0.80);
    }

    #[test]
    fn elevenlabs_v1_tts_timestamps_ws_stream_and_http_loopback_work() {
        reset_mind();
        let mut engine = ElevenLabsEngine::new();
        let dict = engine.add_pronunciation_dictionary(
            "brand_lexicon",
            vec![PronunciationEntry {
                key: "solidSF".to_string(),
                value: "solid S F".to_string(),
                case_sensitive: false,
            }],
        );

        let mut req = TtsRequest::new(
            "[amused] Welcome to solidSF! [short-break] <chuckle> This is ssftalk.",
        );
        req.pronunciation_dictionary_locators.push(PronunciationDictionaryLocator {
            pronunciation_dictionary_id: dict.id.clone(),
            version_id: Some(dict.version_id.clone()),
        });
        req.previous_text = Some("Are you ready for the demo?".to_string());
        req.next_text = Some("Let us inspect the benchmarks next.".to_string());

        let (audio, wav_bytes, ts_resp) = engine.text_to_speech("jesse-1", &req).unwrap();
        assert!(audio.duration_seconds() > 0.5);
        assert!(audio.rms() > 0.01);
        assert_eq!(&wav_bytes[0..4], b"RIFF");
        assert!(!ts_resp.audio_base64.is_empty());
        assert_eq!(ts_resp.alignment.characters.len(), ts_resp.alignment.character_start_times_seconds.len());
        assert!(ts_resp.h2h_verdict.passed_all_gates);

        // Verify ElevenLabs 20-char Rachel UUID alias works seamlessly
        let (rachel_audio, _, _) = engine
            .text_to_speech("21m00Tcm4TlvDq8ikWAM", &TtsRequest::new("Hello from Rachel."))
            .unwrap();
        assert!(rachel_audio.len() > 2048);

        // Verify WebSocket Input Streaming session
        let ws_msgs = vec![
            WsStreamInputMessage {
                text: "Streaming first clause, ".to_string(),
                try_trigger_generation: false,
                flush: false,
                ..Default::default()
            },
            WsStreamInputMessage {
                text: "and finishing the sentence.".to_string(),
                try_trigger_generation: true,
                flush: true,
                ..Default::default()
            },
            WsStreamInputMessage {
                text: String::new(),
                ..Default::default()
            },
        ];
        let ws_out = engine.stream_input_session("jesse-2", &ws_msgs).unwrap();
        assert!(ws_out.len() >= 2);
        assert!(ws_out[0].audio.is_some());
        assert!(ws_out.last().unwrap().is_final);

        // Verify Loopback HTTP/1.1 Server + Client
        let srv = ElevenLabsLoopbackServer::spawn_ephemeral(1).unwrap();
        let http_cli = ElevenLabsHttpClient::new(&srv.base_url, "xi-test-key");
        let http_resp = http_cli
            .post_tts_with_timestamps("jesse-3", &TtsRequest::new("Loopback HTTP parity verified."))
            .unwrap();
        assert!(!http_resp.audio_base64.is_empty());
        assert_eq!(http_resp.character_count, 30);
        srv.join();
    }

    #[test]
    fn voice_cloning_sts_audio_isolation_sfx_scribe_and_wsola_dubbing_work() {
        let mut engine = ElevenLabsEngine::new();
        let (ref_audio, _, _) = engine
            .text_to_speech("jesse-1", &TtsRequest::new("Reference voice sample for instant cloning."))
            .unwrap();

        // 1. Instant Voice Cloning
        let (cloned_voice, clone_receipt) = engine
            .clone_voice(
                "Andy Cloned Studio",
                "Cloned from reference utterance",
                std::slice::from_ref(&ref_audio),
                BTreeMap::new(),
            )
            .unwrap();
        assert!(clone_receipt.extracted_f0_hz >= 85.0);

        // 2. Speech-to-Speech into Baritone (`jesse-3`)
        let sts_receipt = engine
            .convert_speech_to_speech("jesse-3", &ref_audio, None, Some(42))
            .unwrap();
        assert_eq!(sts_receipt.target_voice_id, "jesse-3");
        assert!(!sts_receipt.audio.is_empty());

        // 3. Audio Isolation (add stationary noise to silence + speech and isolate)
        let mut noisy = ref_audio.clone();
        for (i, s) in noisy.samples.iter_mut().enumerate() {
            let hum = 0.04 * ((i as f32) * 0.15).sin();
            *s = (*s + hum).clamp(-1.0, 1.0);
        }
        let iso = engine.isolate_audio(&noisy, 1.0).unwrap();
        assert!(iso.snr_improvement_db > 10.0);
        assert!(iso.wavelet_detail_pruned_ratio > 0.10);

        // 4. Port-Hamiltonian Sound Generation
        let sfx = engine
            .generate_sfx("metallic sci-fi chime resonance", Some(0.6), Some(0.8))
            .unwrap();
        assert!(sfx.rms() > 0.005);

        // 5. Forced Alignment & Scribe ASR
        let scribe = engine
            .transcribe(&ref_audio, Some("Reference voice sample for instant cloning."), true)
            .unwrap();
        assert_eq!(scribe.words.len(), 6);

        // 6. WSOLA Multi-Speaker Timeline Dubbing
        let dub = engine
            .dub_timeline(&[
                DubbingSegment {
                    segment_id: "seg_1".to_string(),
                    voice_id: cloned_voice.voice_id.clone(),
                    source_text: "Hello everyone.".to_string(),
                    target_text: "Bonjour tout le monde, bienvenue.".to_string(),
                    target_language: "fr".to_string(),
                    start_sec: 0.0,
                    end_sec: 0.8,
                },
                DubbingSegment {
                    segment_id: "seg_2".to_string(),
                    voice_id: "jesse-3".to_string(),
                    source_text: "Let us begin.".to_string(),
                    target_text: "Commençons immédiatement.".to_string(),
                    target_language: "fr".to_string(),
                    start_sec: 0.85,
                    end_sec: 1.60,
                },
            ])
            .unwrap();
        assert!(dub.max_timing_drift_ms < 1.0);
    }

    #[test]
    fn j128_codec_duplex_room_and_counterfactual_mechinterp_hold() {
        reset_mind();
        let mut engine = ElevenLabsEngine::new();
        let mut req = TtsRequest::new("Testing 128 kbps companded codec and counterfactual ablation.");
        req.output_format = OutputFormat::J128_24000;
        let (audio, j128_bytes, _) = engine.text_to_speech("jesse-1", &req).unwrap();

        // Exact 15 samples -> 10 bytes ratio
        let expected_blocks = audio.len().div_ceil(codec::J128_BLOCK_SAMPLES);
        assert_eq!(j128_bytes.len(), expected_blocks * codec::J128_BLOCK_BYTES);
        let decoded = codec::decode_audio_output(&j128_bytes, OutputFormat::J128_24000).unwrap();
        assert!(decoded.rms() > 0.01);

        // Counterfactual ablation: disable aperiodic breath + ablate harmonics 2..10
        let ((ablated_audio, _, _), summary) = with_counterfactual_ablation(
            AblationConfig {
                ablated_harmonics: (2..=10).collect(),
                disable_aperiodic_breath: true,
                ..Default::default()
            },
            || engine.text_to_speech("jesse-1", &req).unwrap(),
        );
        assert_ne!(audio.samples, ablated_audio.samples);
        assert!(summary.total_events >= 2);

        // Audio-LLM Frame Token Pruning + 1-Bit SAE Voice Circuit Matcher + 64-D Glass-Box Signature
        let mut sparse_pcm16 = vec![0i16; 24_000]; // 1.0 sec mostly silent with 10% active burst
        for s in &mut sparse_pcm16[2_400..4_800] {
            *s = 8_000;
        }
        let prune = mechinterp::prune_audio_llm_frames(&sparse_pcm16, 480, 250, 8).unwrap();
        assert!(prune.token_savings_ratio >= 0.85);

        let sig = mechinterp::GlassBoxVoiceSignature64D::from_audio(&audio).unwrap();
        assert_eq!(sig.vector.len(), 64);
        assert!((sig.cosine_similarity(&sig) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cloud_ws_and_zobrist_feistel_bounty_seal_hold() {
        let unlocked = cloud_ws::unlock_vault_seal(0x534F_4C49_4453_4621);
        assert!(unlocked.contains("andy@solidsf.com"));
        assert!(unlocked.contains("fly you out to San Francisco"));

        assert_eq!(cloud_ws::purchase_cents_to_microusd(1_000).unwrap(), 10_000_000);
        assert_eq!(cloud_ws::retail_debit_microusd(0.000010, 30), Some(13));

        let server = cloud_ws::SsfTalkLoopbackWsServer::spawn_ephemeral(3).unwrap();
        let mut client =
            cloud_ws::SsfTalkWsClient::connect(&server.ws_url, Some("test-key")).unwrap();

        let r1 = vocoder::dispatch_over_ws(&mut client, "jesse-1", "Hello cloud", 24_000).unwrap();
        assert!(r1.ok);
        assert_eq!(r1.engine, cloud_ws::WsEngine::Articulate150mNeural);
        assert_eq!(r1.credits_charged_microusd, 15);
        assert_eq!(r1.credits_remaining_microusd, 10_000_000 - 15);

        let r2 = hamiltonian::dispatch_over_ws(&mut client, "Cognitive trajectory", 128).unwrap();
        assert!(r2.ok);
        assert_eq!(r2.engine, cloud_ws::WsEngine::HamiltonianProsodyDirector);
        assert_eq!(r2.credits_charged_microusd, 5);
        assert_eq!(r2.credits_remaining_microusd, 10_000_000 - 20);

        let r3 = duplex::dispatch_over_ws(&mut client, "room-alpha", 4).unwrap();
        assert!(r3.ok);
        assert_eq!(r3.engine, cloud_ws::WsEngine::DuplexDiarizer4D);
        assert_eq!(r3.credits_charged_microusd, 10);
        assert_eq!(r3.credits_remaining_microusd, 10_000_000 - 30);

        server.join();

        // Verify fail-closed 402 Payment Required when credits are depleted
        let low_server = cloud_ws::SsfTalkLoopbackWsServer::spawn_with_balance(2, 15).unwrap();
        let mut low_client =
            cloud_ws::SsfTalkWsClient::connect(&low_server.ws_url, Some("low-key")).unwrap();
        let ok_call =
            vocoder::dispatch_over_ws(&mut low_client, "jesse-1", "Last funded utterance", 24_000)
                .unwrap();
        assert_eq!(ok_call.credits_remaining_microusd, 0);
        let depleted =
            vocoder::dispatch_over_ws(&mut low_client, "jesse-1", "Overdrawn utterance", 24_000);
        assert!(depleted.is_err());
        let err = depleted.unwrap_err();
        assert_eq!(err.code, 402);
        assert!(err.message.contains("https://solidsf.com/compute-credits"));
        low_server.join();
    }

    #[test]
    fn audio_studio_prompt_prep_generate_edit_process_and_mcp_tools_hold() {
        let (p1, s1) = studio::prepare_studio_prompt(
            "Lo-fi hip-hop with mellow piano",
            studio::StudioAudioType::Music,
            None,
            Some(85),
            &studio::SfxSoundFields::default(),
        )
        .unwrap();
        assert_eq!(
            p1,
            "TrackType: Music, VocalType: Instrumental, Lo-fi hip-hop with mellow piano. BPM: 85."
        );
        assert_eq!(s1, None);

        let (p2, s2) = studio::prepare_studio_prompt(
            "",
            studio::StudioAudioType::Sfx,
            Some(3.0),
            None,
            &studio::SfxSoundFields::new("glass bottle", "shattering.", "", "concrete floor, dry"),
        )
        .unwrap();
        assert_eq!(
            p2,
            "TrackType: SFX, glass bottle, shattering, concrete floor, dry."
        );
        assert_eq!(s2, Some(3.0));

        let (p3, s3) = studio::prepare_studio_prompt(
            "Ambient drone. Length: 42 seconds.",
            studio::StudioAudioType::Music,
            None,
            None,
            &studio::SfxSoundFields::default(),
        )
        .unwrap();
        assert_eq!(s3, Some(42.0));
        assert!(!p3.contains("Length"));

        let scratch_base = std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("solidsf-scratch")
            .join("ssftalk-unit-studio-out");
        let _ = std::fs::remove_dir_all(&scratch_base);
        let mut st = studio::StudioEngine::new(&scratch_base);
        assert!(!st.unload_models().unloaded);

        let gen = st
            .generate_audio(&studio::GenerateAudioRequest {
                audio_type: studio::StudioAudioType::Sfx,
                duration_seconds: Some(1.0),
                variations: 2,
                seed: Some(7),
                sound_source: "door".to_string(),
                sound_action: "slam".to_string(),
                title: Some("door test".to_string()),
                original_idea: "heavy wooden door slamming".to_string(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(gen.prompt_used, "TrackType: SFX, door, slam.");
        assert!(gen.included_model_load);
        assert_eq!(gen.candidates.len(), 2);
        assert_eq!(gen.candidates[0].seed, 7);
        assert_eq!(gen.candidates[1].seed, 8);
        assert!(gen.candidates[0].peak_dbfs <= -0.95);

        let edit_cont = st
            .edit_audio(&studio::EditAudioRequest {
                input_audio: gen.candidates[0].path.clone(),
                mode: studio::StudioEditMode::Continue,
                prompt: "echoing hallway tail".to_string(),
                audio_type: studio::StudioAudioType::Sfx,
                strength: 0.6,
                start_seconds: None,
                end_seconds: None,
                duration_seconds: Some(2.0),
                overlap_seconds: 0.2,
                variations: 1,
                bpm: None,
                seed: Some(11),
                negative_prompt: None,
                cfg: 1.0,
                steps: 8,
                title: Some("door_extended".to_string()),
                original_idea: String::new(),
                peak_ceiling_db: Some(-1.0),
            })
            .unwrap();
        assert_eq!(edit_cont.seconds, 2.0);
        assert!(edit_cont.region.is_some());

        let proc_res = st
            .process_audio(&studio::ProcessAudioRequest {
                input_audio: edit_cont.candidates[0].path.clone(),
                start_seconds: 0.0,
                end_seconds: None,
                trim_silence: false,
                silence_threshold_db: -50.0,
                mono: true,
                normalize: studio::StudioNormalizeMode::Loudness,
                target_lufs: -16.0,
                peak_ceiling_db: -1.0,
                r#loop: true,
                loop_crossfade_ms: 100.0,
                fade_in_ms: 20.0,
                fade_out_ms: 20.0,
                format: studio::StudioExportFormat::Flac,
                sample_rate: 48_000,
                ogg_quality: 0.6,
                title: Some("door_mastered".to_string()),
            })
            .unwrap();
        assert_eq!(proc_res.channels, 1);
        assert_eq!(proc_res.format, "flac");
        assert_eq!(proc_res.sample_rate, 48_000);
        assert!(proc_res.output_lufs.is_some());
        assert_eq!(proc_res.metadata_sidecar["source"]["seed"], 11);
        assert!(st.unload_models().unloaded);
        let _ = std::fs::remove_dir_all(&scratch_base);
    }
}
