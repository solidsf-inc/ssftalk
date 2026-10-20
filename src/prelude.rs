//! Prelude exports for `ssftalk` and drop-in `ssftalk::elevenlabs` workflows.

pub use crate::cloud_ws::{
    purchase_cents_to_microusd, retail_debit_microusd, unlock_vault_seal, SsfTalkLoopbackWsServer,
    SsfTalkWsClient, WsEngine, WsRpcResponse,
};
pub use crate::codec::{
    decode_audio_output, decode_j128, decode_wav_pcm16, encode_audio_output, encode_j128,
    encode_wav_pcm16, J128_BLOCK_BYTES, J128_BLOCK_SAMPLES,
};
pub use crate::core::{AudioBuffer, Error, OutputFormat, Result, DEFAULT_SAMPLE_RATE};
pub use crate::duplex::{
    decode_jaud_frame, encode_jaud_frame, DuplexRoom, H2HSpeechJury, H2HVerdict, PlaybackState,
    Role, SayMode, Speaker, TranscriptEntry, VoiceMemoryGrant, JAUD_HEADER_BYTES, JAUD_MAGIC,
};
pub use crate::elevenlabs::{
    ElevenLabsEngine, ElevenLabsHttpClient, ElevenLabsLoopbackServer, ModelInfo, TtsRequest,
    TtsWithTimestampsResponse, WsStreamInputMessage, WsStreamOutputMessage,
};
pub use crate::hamiltonian::{
    CharacterAlignment, EmotionAxes8D, FourierPhaseVector, ProsodyPrediction, ProsodyPredictor,
    ThinkingTrajectory, FOURIER_DIM,
};
pub use crate::hardware::{GraceCoherentSlab, HardwareTopology};
pub use crate::mechinterp::{
    mind_summary, prune_audio_llm_frames, reset_mind, snapshot_mind, with_counterfactual_ablation,
    AblationConfig, GlassBoxVoiceSignature64D, MechInterpSummary, SaeVoiceCircuitMatcher,
    VoiceTheoremId,
};
pub use crate::phonetics::{
    g2p_english, Phoneme, PhonemeToken, PronunciationDictionary, PronunciationDictionaryLocator,
    PronunciationEntry,
};
pub use crate::sts_isolate::{
    isolate_voice_audio, speech_to_speech, transcribe_scribe, AudioIsolationReceipt,
    DubbingReceipt, DubbingSegment, SpeechToSpeechReceipt,
};
pub use crate::studio::{
    decode_studio_clip, encode_studio_clip, load_sidecar, prepare_studio_prompt, read_studio_file,
    reserve_output_path, resolve_candidate_seeds, strip_studio_prefix, synthesize_studio_stereo,
    write_sidecar, write_studio_file, AudioSignalStats, EditAudioRequest, EditAudioResponse,
    GenerateAudioRequest, GenerateAudioResponse, LoudnormTelemetry, ProcessAudioRequest,
    ProcessAudioResponse, SfxSoundFields, StudioAudioClip, StudioAudioType, StudioCandidateOutput,
    StudioEditMode, StudioEngine, StudioExportFormat, StudioModelTier, StudioNormalizeMode,
    UnloadModelsResponse, STUDIO_DEFAULT_SAMPLE_RATE, STUDIO_MCP_INSTRUCTIONS,
    STUDIO_VIDEO_SAMPLE_RATE,
};
pub use crate::vocoder::{ArticulateStreamer, AudioChunk, HarmonicIstftVocoder};
pub use crate::voices::{
    VocalTractProfile, Voice, VoiceCloningReceipt, VoiceDesignPreview, VoiceLibrary, VoiceSettings,
    BUILTIN_VOICES,
};
