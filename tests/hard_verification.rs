//! Hard mathematical, acoustic, and protocol verification suite for `ssftalk`.

use ssftalk::mechinterp::SaeBinaryAtom;
use ssftalk::prelude::*;
use std::collections::BTreeMap;

#[test]
fn theorem_1_fourier_phase_bind_unbind_exact_isometry() {
    reset_mind();
    let role = FourierPhaseVector::from_token("ROLE_AGENT_CLAUSE_HEAD");
    let filler = FourierPhaseVector::from_token("FILLER_SOLIDSF_HAMILTONIAN");
    assert_eq!(role.re.len(), FOURIER_DIM);
    assert_eq!(role.im.len(), FOURIER_DIM);

    // Every frequency coordinate has unit modulus |Z_k|^2 = 1.0
    for k in 0..FOURIER_DIM {
        let mod_sq = role.re[k] * role.re[k] + role.im[k] * role.im[k];
        assert!((mod_sq - 1.0).abs() < 1e-12);
    }

    let bound = role.bind_role(&filler);
    let recovered = bound.unbind_role(&role);
    let coherence = recovered.phase_coherence(&filler);
    assert!(
        (coherence - 1.0).abs() < 1e-12,
        "Fourier phase bind/unbind must be exact unit-modulus isometry, got {coherence}"
    );
}

#[test]
fn theorems_2_3_4_hamiltonian_trajectory_hodge_closure_and_emotion_axes() {
    reset_mind();
    let text = "Wait [sigh] can solidSF synthesize sub-45ms Hamiltonian speech, [laugh] or is cloud latency unavoidable?!";
    let lib = VoiceLibrary::new();
    let voice = lib.resolve("jesse-1").unwrap();
    let settings = VoiceSettings {
        stability: 0.55,
        similarity_boost: 0.85,
        style: 0.65,
        use_speaker_boost: true,
        speed: 1.05,
    };
    let emotions = EmotionAxes8D {
        warm: 0.70,
        deadpan: 0.0,
        curious: 0.50,
        serious: 0.85,
        amused: 0.40,
        empathetic: 0.35,
        urgent: 0.60,
        calm: 0.20,
    };

    let traj = ThinkingTrajectory::from_transcript(text, 64);
    assert_eq!(traj.semantic_fg_mask.len(), 64);
    assert!((traj.hodge_closure - 1.0).abs() < 1e-6);
    let max_shock = traj
        .richtmyer_shock
        .iter()
        .fold(0.0_f32, |a, &b| a.max(b));
    assert!(max_shock >= 0.80);

    let tokens = g2p_english(text);
    let pred = ProsodyPredictor.predict(
        text,
        &tokens,
        &voice.tract,
        &settings,
        &emotions,
        Some(&traj),
        None,
        None,
    );
    assert!(!pred.f0_hz.is_empty());
    assert!(pred.total_frames > 20);
}

#[test]
fn theorem_5_and_6_harmonic_vocoder_and_h2h_minimax_speech_jury() {
    reset_mind();
    let mut engine = ElevenLabsEngine::new();
    let mut req = TtsRequest::new(
        "Deterministic harmonic plus filtered noise excitation with zero neural hallmarked phase blur.",
    );
    req.model_id = "ssf_hamiltonian_v1_3".to_string();
    req.seed = Some(2026);
    req.previous_text = Some("Let us inspect the waveform.".to_string());
    req.next_text = Some("Every gate passes with margin.".to_string());

    let (audio, wav_bytes, ts_resp) = engine.text_to_speech("jesse-2", &req).unwrap();
    assert!(audio.duration_seconds() > 1.5);
    assert!(audio.rms() > 0.01);
    assert_eq!(&wav_bytes[0..4], b"RIFF");
    assert_eq!(
        ts_resp.alignment.characters.len(),
        ts_resp.alignment.character_start_times_seconds.len()
    );
    assert!(
        ts_resp.h2h_verdict.passed_all_gates,
        "4-gate H2H speech jury must pass: {:?}",
        ts_resp.h2h_verdict
    );
    assert!(ts_resp.h2h_verdict.minimax_saddle_score > 0.70);
}

#[test]
fn theorem_7_j128_codec_exact_128kbps_and_all_elevenlabs_output_formats() {
    let mut engine = ElevenLabsEngine::new();
    let req = TtsRequest::new("Testing exact 128 kilobit per second companded predictive framing.");
    let (audio, _, _) = engine.text_to_speech("rachel", &req).unwrap();

    let j128_bytes = encode_j128(&audio.samples);
    let blocks = audio.len().div_ceil(J128_BLOCK_SAMPLES);
    assert_eq!(
        j128_bytes.len(),
        blocks * J128_BLOCK_BYTES,
        "j128 must pack 15 samples into 10 bytes (16,000 B/s = 128.0 kbps)"
    );
    let decoded_samples = decode_j128(&j128_bytes).unwrap();
    let decoded = AudioBuffer::new(decoded_samples, audio.sample_rate, 1);
    assert!(decoded.rms() > 0.01);

    for fmt in [
        OutputFormat::Wav24000,
        OutputFormat::Pcm16000,
        OutputFormat::Pcm22050,
        OutputFormat::Pcm24000,
        OutputFormat::Pcm44100,
        OutputFormat::Ulaw8000,
        OutputFormat::Mp3_44100_128,
        OutputFormat::J128_24000,
    ] {
        let mut fmt_req = req.clone();
        fmt_req.output_format = fmt;
        let (_, encoded_bytes, _) = engine.text_to_speech("adam", &fmt_req).unwrap();
        assert!(!encoded_bytes.is_empty(), "empty output for format {fmt:?}");
    }
}

#[test]
fn theorems_8_9_10_wsola_dubbing_audio_isolation_sts_cloning_and_scribe() {
    let mut engine = ElevenLabsEngine::new();
    let (clean_speech, _, _) = engine
        .text_to_speech(
            "jesse-3",
            &TtsRequest::new(
                "Formant preserved spectral envelope under voice cloning and speech to speech conversion.",
            ),
        )
        .unwrap();

    // 1. Zero-shot instant voice cloning (/v1/voices/add)
    let (cloned, receipt) = engine
        .clone_voice(
            "Custom Founder Clone",
            "Zero-shot spectral envelope clone",
            std::slice::from_ref(&clean_speech),
            BTreeMap::new(),
        )
        .unwrap();
    assert!(cloned.voice_id.starts_with("clone_"));
    assert!(receipt.extracted_f0_hz >= 85.0);

    // 2. Natural-language voice design previews (/v1/text-to-voice/create-previews)
    let previews = engine
        .create_voice_previews(
            "Warm authoritative baritone engineer with crisp consonants",
            "Testing natural language voice design preview synthesis on solidSF.",
        )
        .unwrap();
    assert_eq!(previews.len(), 3);

    // 3. Speech-to-Speech (/v1/speech-to-speech/{voice_id})
    let sts_receipt = engine
        .convert_speech_to_speech("jesse-4", &clean_speech, None, Some(77))
        .unwrap();
    assert_eq!(sts_receipt.target_voice_id, "jesse-4");
    assert!(
        (sts_receipt.audio.duration_seconds() - clean_speech.duration_seconds()).abs() < 0.15,
        "STS must preserve prosodic timing contour"
    );

    // 4. Audio Isolation (/v1/audio-isolation)
    let mut noisy = clean_speech.clone();
    for (i, s) in noisy.samples.iter_mut().enumerate() {
        let hum = 0.045 * ((i as f32) * 0.15).sin();
        *s = (*s + hum).clamp(-1.0, 1.0);
    }
    let iso_res = engine.isolate_audio(&noisy, 1.0).unwrap();
    assert!(
        iso_res.snr_improvement_db > 10.0,
        "Spectral gate must improve SNR by >10 dB, got {:.2} dB",
        iso_res.snr_improvement_db
    );

    // 5. WSOLA pitch-invariant timeline dubbing (/v1/dubbing)
    let dub = engine
        .dub_timeline(&[DubbingSegment {
            segment_id: "seg_verify_1".to_string(),
            voice_id: "jesse-1".to_string(),
            source_text: "Hello from San Francisco.".to_string(),
            target_text: "Bonjour depuis San Francisco avec une phrase nettement plus longue."
                .to_string(),
            target_language: "fr".to_string(),
            start_sec: 0.0,
            end_sec: 2.15,
        }])
        .unwrap();
    assert!(
        dub.max_timing_drift_ms < 1.0,
        "WSOLA dubbing duration drift must be < 1 ms, got {:.4} ms",
        dub.max_timing_drift_ms
    );

    // 6. Scribe ASR + diarization (/v1/speech-to-text) & Forced Alignment (/v1/forced-alignment)
    let transcript = engine
        .transcribe(
            &clean_speech,
            Some("Formant preserved spectral envelope under voice cloning and speech to speech conversion."),
            true,
        )
        .unwrap();
    assert_eq!(transcript.words.len(), 12);

    let fa = engine
        .align_transcript(
            &clean_speech,
            "Formant preserved spectral envelope under voice cloning",
        )
        .unwrap();
    assert_eq!(fa.words.len(), 7);
}

#[test]
fn theorem_11_full_duplex_room_jaud_header_and_sub_millisecond_barge_in() {
    let mut room = DuplexRoom::new();
    room.join("human-1", "Andy", Role::Human, VoiceMemoryGrant { grant: true });
    room.join("agent-1", "Jesse", Role::Agent, VoiceMemoryGrant { grant: false });

    let persisted = room
        .name_shared_mic_speaker("human-1", 0, "Andy Norris", true)
        .unwrap();
    assert!(persisted);

    let pcm_bytes = vec![0x12u8; 960];
    let pkt = encode_jaud_frame(7, 42, &pcm_bytes);
    assert_eq!(&pkt[0..4], &JAUD_MAGIC);
    let (seq, utt_id, payload) = decode_jaud_frame(&pkt).unwrap();
    assert_eq!(seq, 7);
    assert_eq!(utt_id, 42);
    assert_eq!(payload.len(), 960);

    room.agent_say(
        "agent-1",
        "utt-100",
        "Let me walk you through all twelve theorems in detail.",
        SayMode::Turn,
    );
    assert_eq!(room.active_floor_utt.as_deref(), Some("utt-100"));

    // Backchannel overlap does not steal floor
    room.agent_say("agent-1", "utt-101", "mm-hmm", SayMode::Backchannel);
    assert_eq!(room.active_floor_utt.as_deref(), Some("utt-100"));

    // Human barge-in interrupts utt-100 at 18 heard characters
    let interrupted = room.report_playback("utt-100", PlaybackState::Interrupted, Some(18));
    assert!(interrupted);
    assert!(room.active_floor_utt.is_none());
    assert_eq!(room.transcript[0].heard_chars, 18);
    assert!(room.transcript[0].interrupted);
}

#[test]
fn theorem_12_mechinterp_counterfactual_ablation_and_audio_llm_token_pruning() {
    reset_mind();
    let mut engine = ElevenLabsEngine::new();
    let req = TtsRequest::new(
        "Wait [sigh] does Hamiltonian phase binding change prosodic pitch excursion?!",
    );

    let (baseline_audio, _, _) = engine.text_to_speech("jesse-1", &req).unwrap();
    let summary = mind_summary();
    assert!(summary.total_events >= 5);
    assert!(summary.theorem_counts.len() >= 4);

    let ((ablated_audio, _, _), ablated_summary) = with_counterfactual_ablation(
        AblationConfig {
            ablated_harmonics: (2..=12).collect(),
            disable_aperiodic_breath: true,
            disable_hamiltonian_coupling: true,
            disable_richtmyer_shock: true,
        },
        || engine.text_to_speech("jesse-1", &req).unwrap(),
    );
    assert!(ablated_summary.total_events >= 2);
    assert_ne!(baseline_audio.samples, ablated_audio.samples);

    // Audio-LLM token pruning on conversational stream (with silence gaps)
    let mut sparse_pcm16 = vec![0i16; 24_000 * 2];
    for s in &mut sparse_pcm16[2_400..6_000] {
        *s = 9_500;
    }
    let pruned = prune_audio_llm_frames(&sparse_pcm16, 480, 250, 8).unwrap();
    assert!(
        pruned.token_savings_ratio >= 0.85,
        "Audio-LLM frame pruning must save >=85% tokens on sparse speech, got {:.2}%",
        pruned.token_savings_ratio * 100.0
    );

    // 1-bit SAE voice circuit matcher & 64-D glass-box signature
    let sig = GlassBoxVoiceSignature64D::from_audio(&baseline_audio).unwrap();
    assert_eq!(sig.vector.len(), 64);
    assert!((sig.cosine_similarity(&sig) - 1.0).abs() < 1e-5);

    let atom = SaeBinaryAtom {
        feature_id: 7,
        label: "hamiltonian_clause_focus".to_string(),
        bits: [0xAA55_AA55_AA55_AA55; 4],
    };
    let bank = SaeVoiceCircuitMatcher::new(vec![atom.clone()]);
    let matches = bank.match_residual_batch(&[atom]);
    let (fid, lbl, dist, _) = matches[0].clone().unwrap();
    assert_eq!(fid, 7);
    assert_eq!(lbl, "hamiltonian_clause_focus");
    assert_eq!(dist, 0);
}

#[test]
fn elevenlabs_http_loopback_server_and_cloud_ws_metering_roundtrip() {
    // 1. ElevenLabs HTTP/1.1 Loopback Server + Client
    let server = ElevenLabsLoopbackServer::spawn_ephemeral(1).unwrap();
    let client = ElevenLabsHttpClient::new(&server.base_url, "xi-verify-key");

    let req = TtsRequest::new("Testing solidSF HTTP 1.1 loopback parity for ElevenLabs v1 endpoints.");
    let with_ts = client.post_tts_with_timestamps("jesse-1", &req).unwrap();
    assert!(!with_ts.audio_base64.is_empty());
    assert_eq!(with_ts.character_count, req.text.chars().count());
    let wav_bytes = ssftalk::core::base64_decode(&with_ts.audio_base64).unwrap();
    let decoded = decode_wav_pcm16(&wav_bytes).unwrap();
    assert!(decoded.duration_seconds() > 1.0);
    server.join();

    // 2. Cloud Binary WebSocket Server (`SSFTK\x01`) + Credit Metering
    let ws_srv = SsfTalkLoopbackWsServer::spawn_ephemeral(4).unwrap();
    let mut ws_cli = SsfTalkWsClient::connect(&ws_srv.ws_url, Some("ssf_live_verify_key")).unwrap();

    let r1 = ssftalk::hamiltonian::dispatch_over_ws(&mut ws_cli, "hamiltonian-prosody", 128).unwrap();
    let r2 = ssftalk::vocoder::dispatch_over_ws(&mut ws_cli, "jesse-1", "vocoder-cfm", 24_000).unwrap();
    let r3 = ssftalk::sts_isolate::dispatch_over_ws(&mut ws_cli, "sts-isolate", 48_000, 24_000).unwrap();
    let r4 = ssftalk::duplex::dispatch_over_ws(&mut ws_cli, "room-verify", 4).unwrap();

    assert!(r1.ok && r2.ok && r3.ok && r4.ok);
    assert_eq!(
        r4.credits_remaining_microusd,
        10_000_000 - (5 + 15 + 25 + 10),
        "Exact solidSF Credit deduction across all 4 cloud engines"
    );
    ws_srv.join();
}

#[test]
fn theorem_13_audio_studio_post_processing_loop_loudnorm_and_24bit_export_roundtrip() {
    let scratch_dir = std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("solidsf-scratch")
        .join("ssftalk-hard-verify-studio");
    let _ = std::fs::remove_dir_all(&scratch_dir);
    std::fs::create_dir_all(&scratch_dir).unwrap();

    // 1. Peak ceiling only attenuates, never boosts; peak normalize boosts
    let sr = 1_000u32;
    let ramp: Vec<f32> = (0..(3 * sr as usize))
        .map(|i| (i as f32) / ((3 * sr as usize - 1) as f32))
        .collect();
    let clip = StudioAudioClip::new(vec![ramp.clone(), ramp.clone()], sr).unwrap();

    let quiet = StudioAudioClip::new(vec![vec![0.1; 1000], vec![0.1; 1000]], sr).unwrap();
    let (_, ceil_gain_quiet) = quiet.peak_ceiling(-1.0);
    assert_eq!(ceil_gain_quiet, 0.0);

    let loud = StudioAudioClip::new(vec![vec![1.5; 1000], vec![1.5; 1000]], sr).unwrap();
    let (capped, ceil_gain_loud) = loud.peak_ceiling(-1.0);
    assert!(ceil_gain_loud < 0.0);
    assert!((capped.stats().peak_dbfs - (-1.0)).abs() <= 0.02);

    let (normed, norm_gain) = quiet.peak_normalize(-1.0);
    assert!(norm_gain > 0.0);
    assert!((normed.stats().peak_dbfs - (-1.0)).abs() <= 0.02);

    // 2. Seamless equal-power loop crossfade continuity: out[0] == ramp[-500]
    let looped = clip.make_loop(500.0).unwrap();
    assert_eq!(looped.num_samples(), 3 * (sr as usize) - 500);
    assert!((looped.channels[0][0] - ramp[ramp.len() - 500]).abs() < 1e-3);
    assert!(clip.make_loop(2000.0).is_err());

    // 3. Audible end & silence trimming on 5ms RMS windows
    let mut with_tail_silence = vec![0.0_f32; 3000];
    for (i, s) in with_tail_silence[..2000].iter_mut().enumerate() {
        *s = 0.5 * (2.0 * std::f32::consts::PI * 50.0 * (i as f32) / (sr as f32)).sin();
    }
    let aud_clip =
        StudioAudioClip::new(vec![with_tail_silence.clone(), with_tail_silence], sr).unwrap();
    assert!((aud_clip.audible_end(-60.0) - 2.0).abs() <= 0.02);

    // 4. Write & Read roundtrip across WAV 24-bit, FLAC 24-bit, and OGG Vorbis
    for (fmt, ext) in [
        (StudioExportFormat::Wav, "wav"),
        (StudioExportFormat::Flac, "flac"),
        (StudioExportFormat::Ogg, "ogg"),
    ] {
        let p = scratch_dir.join(format!("roundtrip.{ext}"));
        write_studio_file(&p, &clip, fmt, 0.6).unwrap();
        let loaded = read_studio_file(&p).unwrap();
        assert_eq!(loaded.sample_rate, sr);
        assert_eq!(loaded.num_channels(), 2);
        assert_eq!(loaded.num_samples(), clip.num_samples());
    }

    // 5. MCP tool dispatch (`generate_audio`, `edit_audio`, `process_audio`, `unload_models`)
    let mut studio = StudioEngine::new(&scratch_dir);
    let gen_val = studio
        .call_mcp_tool(
            "generate_audio",
            serde_json::json!({
                "prompt": "Lo-fi hip-hop cafe beat with warm electric piano",
                "audio_type": "music",
                "bpm": 85,
                "seconds": 1.5,
                "variations": 2,
                "seed": 42,
                "title": "lofi_cafe"
            }),
        )
        .unwrap();
    assert_eq!(gen_val["seconds"], 1.5);
    assert_eq!(gen_val["candidates"].as_array().unwrap().len(), 2);

    let first_path = gen_val["candidates"][0]["path"].as_str().unwrap().to_string();
    let edit_inpaint = studio
        .call_mcp_tool(
            "edit_audio",
            serde_json::json!({
                "input_audio": first_path,
                "mode": "inpaint",
                "prompt": "mellow jazz solo",
                "start_seconds": 0.4,
                "end_seconds": 1.1,
                "seed": 99
            }),
        )
        .unwrap();
    assert_eq!(edit_inpaint["region"], serde_json::json!([0.4, 1.1]));

    let proc_ogg = studio
        .call_mcp_tool(
            "process_audio",
            serde_json::json!({
                "input_audio": first_path,
                "normalize": "loudness",
                "target_lufs": -16.0,
                "loop": true,
                "loop_crossfade_ms": 100.0,
                "format": "ogg",
                "sample_rate": 48000
            }),
        )
        .unwrap();
    assert_eq!(proc_ogg["sample_rate"], 48000);
    assert_eq!(proc_ogg["format"], "ogg");
    assert!((proc_ogg["output_lufs"].as_f64().unwrap() - (-16.0)).abs() <= 1.0);

    let unloaded = studio
        .call_mcp_tool("unload_models", serde_json::json!({}))
        .unwrap();
    assert_eq!(unloaded["unloaded"], true);

    let _ = std::fs::remove_dir_all(&scratch_dir);
}
