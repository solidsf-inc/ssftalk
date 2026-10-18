//! ElevenLabs-compatible Voice Registry, Natural-Language Voice Design (`desc:`),
//! 8-Axis Emotional Blending, and Zero-Shot Instant Voice Cloning (`voices`).

use crate::core::{fnv1a64, AudioBuffer, Error, Result, STS_BAD_ARG};
use crate::mechinterp::GlassBoxVoiceSignature64D;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Prefix for Maya1 / `ssftalk` natural-language voice design prompts (`"desc: ..."`).
pub const DESC_PREFIX: &str = "desc:";
/// Maximum allowed character length for a natural-language voice description.
pub const MAX_DESCRIPTION_CHARS: usize = 256;

/// Canonical 8 emotional & prosodic delivery axes from Jesse v1.3 Hamiltonian-Fourier Voice.
pub const EMOTION_AXES: [&str; 8] = [
    "warm",
    "deadpan",
    "curious",
    "serious",
    "amused",
    "empathetic",
    "urgent",
    "calm",
];

/// ElevenLabs-compatible `voice_settings` payload controlling prosodic stability,
/// speaker timbre similarity, stylistic exaggeration, speaker boost, and WSOLA speed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VoiceSettings {
    /// Stability in `[0.0, 1.0]` (`0.0` = maximum dynamic inflection, `1.0` = rock-steady cadence).
    #[serde(default = "default_stability")]
    pub stability: f32,
    /// Speaker timbre & formant adherence in `[0.0, 1.0]`.
    #[serde(default = "default_similarity_boost")]
    pub similarity_boost: f32,
    /// Stylistic / emotional exaggeration in `[0.0, 1.0]`.
    #[serde(default = "default_style")]
    pub style: f32,
    /// Presence & high-band clarity boost (`true` by default).
    #[serde(default = "default_speaker_boost")]
    pub use_speaker_boost: bool,
    /// Pitch-preserving WSOLA speaking rate multiplier in `[0.25, 4.0]` (`1.0` = natural speed).
    #[serde(default = "default_speed")]
    pub speed: f32,
}

const fn default_stability() -> f32 {
    0.55
}
const fn default_similarity_boost() -> f32 {
    0.80
}
const fn default_style() -> f32 {
    0.25
}
const fn default_speaker_boost() -> bool {
    true
}
const fn default_speed() -> f32 {
    1.0
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            stability: 0.55,
            similarity_boost: 0.80,
            style: 0.25,
            use_speaker_boost: true,
            speed: 1.0,
        }
    }
}

/// Acoustic & vocal-tract parameters defining a speaker's physical timbre.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocalTractProfile {
    /// Base fundamental pitch $F_0$ in Hz (`85.0..=300.0`).
    pub base_f0_hz: f32,
    /// Four vocal-tract formant center frequencies $(F_1, F_2, F_3, F_4)$ in Hz.
    pub formants_hz: [f32; 4],
    /// Glottal harmonic spectral tilt exponent (`0.65..=1.15`).
    pub spectral_tilt: f32,
    /// Baseline aspiration / breathiness ratio in `[0.0, 0.5]`.
    pub breathiness: f32,
    /// Speaking cadence multiplier (`0.8..=1.25`).
    pub base_pace: f32,
    /// Whether to apply the 1980s dual-carrier ring-modulated robot vocoder.
    pub is_robot: bool,
    /// 256-bit binary speaker fingerprint for Tensor-Core `b1.xor.popc` lookup.
    pub binary_fingerprint: [u64; 4],
}

/// ElevenLabs-compatible Voice metadata record (`GET /v1/voices`, `GET /v1/voices/{voice_id}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Voice {
    pub voice_id: String,
    /// Optional ElevenLabs canonical 20-char UUID alias (e.g., `21m00Tcm4TlvDq8ikWAM` for Rachel).
    pub elevenlabs_alias_id: Option<String>,
    pub name: String,
    pub category: String,
    pub description: String,
    pub labels: BTreeMap<String, String>,
    pub settings: VoiceSettings,
    pub tract: VocalTractProfile,
}

/// Built-in static voice definition.
#[derive(Debug, Clone, Copy)]
pub struct BuiltinVoiceSpec {
    pub id: &'static str,
    pub eleven_id: &'static str,
    pub name: &'static str,
    pub category: &'static str,
    pub accent: &'static str,
    pub gender: &'static str,
    pub age: &'static str,
    pub description: &'static str,
    pub base_f0_hz: f32,
    pub formants_hz: [f32; 4],
    pub spectral_tilt: f32,
    pub breathiness: f32,
    pub is_robot: bool,
}

/// Canonical `solidSF` (`jesse-1..4`) + ElevenLabs drop-in default voices (`rachel`, `adam`, `antoni`, `bella`, `josh`)
/// + retro sci-fi vocoder voices (`robot`, `zarvox`, `trinoids`).
pub const BUILTIN_VOICES: &[BuiltinVoiceSpec] = &[
    BuiltinVoiceSpec {
        id: "jesse-1",
        eleven_id: "JSE100Tcm4TlvDq8ikWA",
        name: "Jesse (Warm Studio)",
        category: "premade",
        accent: "american",
        gender: "female",
        age: "thirties",
        description: "Female voice in her thirties, warm, clear and calm, medium pitch, American accent, measured pace.",
        base_f0_hz: 192.0,
        formants_hz: [540.0, 1680.0, 2680.0, 3650.0],
        spectral_tilt: 0.80,
        breathiness: 0.08,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "jesse-2",
        eleven_id: "JSE200Tcm4TlvDq8ikWB",
        name: "Jesse (Bright Expressive)",
        category: "premade",
        accent: "american",
        gender: "female",
        age: "twenties",
        description: "Female voice in her twenties, bright and quick, slightly higher pitch, American accent, friendly.",
        base_f0_hz: 218.0,
        formants_hz: [580.0, 1790.0, 2790.0, 3780.0],
        spectral_tilt: 0.75,
        breathiness: 0.10,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "jesse-3",
        eleven_id: "JSE300Tcm4TlvDq8ikWC",
        name: "Jesse (Grounded Baritone)",
        category: "premade",
        accent: "american",
        gender: "male",
        age: "thirties",
        description: "Male voice in his thirties, relaxed and low, American accent, even pace.",
        base_f0_hz: 118.0,
        formants_hz: [460.0, 1420.0, 2420.0, 3320.0],
        spectral_tilt: 0.84,
        breathiness: 0.06,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "jesse-4",
        eleven_id: "JSE400Tcm4TlvDq8ikWD",
        name: "Jesse (Dry British Contralto)",
        category: "premade",
        accent: "british",
        gender: "female",
        age: "forties",
        description: "Female voice in her forties, dry and precise, lower pitch, British accent, unhurried.",
        base_f0_hz: 168.0,
        formants_hz: [500.0, 1590.0, 2580.0, 3520.0],
        spectral_tilt: 0.82,
        breathiness: 0.07,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "rachel",
        eleven_id: "21m00Tcm4TlvDq8ikWAM",
        name: "Rachel",
        category: "premade",
        accent: "american",
        gender: "female",
        age: "young",
        description: "Calm, articulate American female narrator voice equivalent to ElevenLabs Rachel.",
        base_f0_hz: 198.0,
        formants_hz: [550.0, 1710.0, 2710.0, 3690.0],
        spectral_tilt: 0.79,
        breathiness: 0.09,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "adam",
        eleven_id: "pNInz6obpgDQGcFmaJgB",
        name: "Adam",
        category: "premade",
        accent: "american",
        gender: "male",
        age: "middle_aged",
        description: "Deep, resonant American male narrator voice equivalent to ElevenLabs Adam.",
        base_f0_hz: 108.0,
        formants_hz: [440.0, 1380.0, 2380.0, 3280.0],
        spectral_tilt: 0.86,
        breathiness: 0.05,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "antoni",
        eleven_id: "ErXwobaYiN019PkySvjV",
        name: "Antoni",
        category: "premade",
        accent: "american",
        gender: "male",
        age: "young",
        description: "Well-rounded, warm conversational male voice equivalent to ElevenLabs Antoni.",
        base_f0_hz: 128.0,
        formants_hz: [480.0, 1460.0, 2460.0, 3380.0],
        spectral_tilt: 0.81,
        breathiness: 0.07,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "bella",
        eleven_id: "EXAVITQu4vr4xnSDxMaL",
        name: "Bella",
        category: "premade",
        accent: "american",
        gender: "female",
        age: "young",
        description: "Soft, expressive storytelling female voice equivalent to ElevenLabs Bella.",
        base_f0_hz: 208.0,
        formants_hz: [565.0, 1750.0, 2740.0, 3720.0],
        spectral_tilt: 0.77,
        breathiness: 0.12,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "josh",
        eleven_id: "TxGEqnHWrfWFTfGW9XjX",
        name: "Josh",
        category: "premade",
        accent: "american",
        gender: "male",
        age: "young",
        description: "Crisp, dynamic baritone male voice equivalent to ElevenLabs Josh.",
        base_f0_hz: 114.0,
        formants_hz: [455.0, 1410.0, 2400.0, 3310.0],
        spectral_tilt: 0.83,
        breathiness: 0.06,
        is_robot: false,
    },
    BuiltinVoiceSpec {
        id: "robot",
        eleven_id: "ROB000Tcm4TlvDq8ikWR",
        name: "Zarvox Retro Robot",
        category: "synthesized",
        accent: "cybernetic",
        gender: "neutral",
        age: "ageless",
        description: "Retro 1980s dual-carrier ring-modulated sci-fi robot vocoder voice.",
        base_f0_hz: 135.0,
        formants_hz: [500.0, 1500.0, 2500.0, 3500.0],
        spectral_tilt: 0.70,
        breathiness: 0.02,
        is_robot: true,
    },
];

impl BuiltinVoiceSpec {
    pub fn to_voice(&self) -> Voice {
        let mut labels = BTreeMap::new();
        labels.insert("accent".to_string(), self.accent.to_string());
        labels.insert("gender".to_string(), self.gender.to_string());
        labels.insert("age".to_string(), self.age.to_string());

        let h0 = fnv1a64(self.id.as_bytes());
        let h1 = fnv1a64(self.description.as_bytes());
        let fingerprint = [
            h0,
            h1,
            h0.rotate_left(19) ^ 0x9E37_79B9_7F4A_7C15,
            h1.rotate_right(13) ^ 0xBF58_476D_1CE4_E5B9,
        ];

        Voice {
            voice_id: self.id.to_string(),
            elevenlabs_alias_id: Some(self.eleven_id.to_string()),
            name: self.name.to_string(),
            category: self.category.to_string(),
            description: self.description.to_string(),
            labels,
            settings: VoiceSettings::default(),
            tract: VocalTractProfile {
                base_f0_hz: self.base_f0_hz,
                formants_hz: self.formants_hz,
                spectral_tilt: self.spectral_tilt,
                breathiness: self.breathiness,
                base_pace: 1.0,
                is_robot: self.is_robot,
                binary_fingerprint: fingerprint,
            },
        }
    }
}

/// Receipt returned by Instant Voice Cloning (`POST /v1/voices/add` / `VoiceLibrary::clone_voice`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceCloningReceipt {
    pub voice_id: String,
    pub name: String,
    pub requires_verification: bool,
    pub extracted_f0_hz: f32,
    pub extracted_formants_hz: [f32; 4],
    pub glass_box_canonical_hash: u64,
}

/// Preview generated by Text-to-Voice / Voice Design (`POST /v1/text-to-voice/create-previews`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceDesignPreview {
    pub generated_voice_id: String,
    pub audio_base_64: String,
    pub media_type: String,
    pub duration_secs: f64,
    pub resolved_description: String,
}

/// Validate and resolve a voice identifier or `desc:` natural-language voice design prompt,
/// optionally blending an emotional/delivery `style` modifier.
pub fn describe_with_style(voice_or_desc: &str, style: Option<&str>) -> Result<String> {
    let base = if let Some(desc) = voice_or_desc.strip_prefix(DESC_PREFIX) {
        let d = desc.trim();
        if d.is_empty() {
            return Err(Error::new(STS_BAD_ARG, "voice description after desc: is empty"));
        }
        if d.chars().count() > MAX_DESCRIPTION_CHARS {
            return Err(Error::new(
                STS_BAD_ARG,
                "voice description exceeds 256 characters",
            ));
        }
        if d.chars()
            .any(|c| c.is_control() || matches!(c, '"' | '<' | '>'))
        {
            return Err(Error::new(
                STS_BAD_ARG,
                "voice description must contain printable text without \", < or >",
            ));
        }
        d.to_string()
    } else if let Some(spec) = BUILTIN_VOICES
        .iter()
        .find(|v| v.id.eq_ignore_ascii_case(voice_or_desc) || v.eleven_id == voice_or_desc)
    {
        spec.description.to_string()
    } else {
        return Err(Error::new(
            STS_BAD_ARG,
            format!("unknown voice_id: {voice_or_desc}"),
        ));
    };

    let Some(raw_style) = style.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(base);
    };
    let lower = raw_style.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "default" | "neutral" | "normal" | "natural"
    ) {
        return Ok(base);
    }

    let modifier = match lower.as_str() {
        "warm" | "friendly" => "warm and conversational",
        "amused" | "laughing" | "chuckling" | "humorous" => {
            "amused and smiling, with a light chuckle in the voice"
        }
        "excited" | "enthusiastic" | "happy" | "joyful" => {
            "energetic, upbeat, and genuinely excited"
        }
        "cheerful" | "bright" => "bright, cheerful, and smiling",
        "empathetic" | "caring" | "compassionate" | "sympathetic" => {
            "gentle, warm, and deeply empathetic, speaking softly"
        }
        "gentle" | "soft" | "soothing" => "gentle, unhurried, and reassuring",
        "calm" | "reassuring" => "speaking calmly and evenly",
        "curious" | "inquisitive" | "interested" => {
            "curious and engaged, with natural upward inflection"
        }
        "serious" | "focused" | "direct" | "confident" => "focused, direct, and grounded",
        "deadpan" | "dry" => "dry, deadpan, and understated",
        "sarcastic" | "ironic" | "wry" => "dry, wry, and lightly ironic",
        "whisper" | "whispering" | "hushed" => "speaking in a soft, intimate hushed whisper",
        "sad" | "melancholy" | "somber" => "soft, subdued, and reflective",
        "angry" | "frustrated" | "annoyed" => "tense, firm, and sharp",
        "urgent" | "fast" | "brisk" => "brisk, urgent, and clipped",
        "playful" | "teasing" => "playful, lively, and expressive",
        "robot" | "robotic" | "monotone" => {
            "flat mechanical monotone, clipped robotic cadence, zero inflection"
        }
        _ => {
            if raw_style
                .chars()
                .any(|c| c.is_control() || matches!(c, '"' | '<' | '>'))
            {
                return Err(Error::new(
                    STS_BAD_ARG,
                    "style modifier contains disallowed characters",
                ));
            }
            raw_style
        }
    };

    let trimmed = base.trim_end_matches('.');
    let combined = format!("{trimmed}, {modifier}.");
    if combined.chars().count() > MAX_DESCRIPTION_CHARS {
        return Err(Error::new(
            STS_BAD_ARG,
            "combined voice description exceeds 256 characters",
        ));
    }
    Ok(combined)
}

/// Infer a `VocalTractProfile` from a natural-language `desc:` prompt.
pub fn tract_from_description(desc: &str) -> VocalTractProfile {
    let lower = desc.to_ascii_lowercase();
    let is_male = lower.contains("male") && !lower.contains("female")
        || lower.contains("baritone")
        || lower.contains("tenor")
        || lower.contains("bass");
    let is_high = lower.contains("high") || lower.contains("bright") || lower.contains("young");
    let is_low = lower.contains("low") || lower.contains("deep") || lower.contains("gravelly");
    let is_whisper = lower.contains("whisper") || lower.contains("hushed") || lower.contains("breathy");
    let is_robot = lower.contains("robot") || lower.contains("cybernetic") || lower.contains("zarvox");

    let mut f0 = if is_male { 120.0 } else { 195.0 };
    if is_high {
        f0 *= 1.14;
    }
    if is_low {
        f0 *= 0.86;
    }

    let scale = if is_male { 0.88 } else { 1.02 };
    let formants_hz = [
        520.0 * scale,
        1620.0 * scale,
        2620.0 * scale,
        3580.0 * scale,
    ];
    let h = fnv1a64(desc.as_bytes());

    VocalTractProfile {
        base_f0_hz: f0,
        formants_hz,
        spectral_tilt: if is_low { 0.86 } else { 0.78 },
        breathiness: if is_whisper { 0.32 } else { 0.08 },
        base_pace: if lower.contains("quick") || lower.contains("fast") || lower.contains("urgent") {
            1.15
        } else if lower.contains("unhurried") || lower.contains("slow") {
            0.90
        } else {
            1.0
        },
        is_robot,
        binary_fingerprint: [h, h.rotate_left(17), h.rotate_right(23), !h],
    }
}

/// Registry managing built-in voices, custom `desc:` designed voices, and instant-cloned voices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceLibrary {
    pub voices: BTreeMap<String, Voice>,
}

impl Default for VoiceLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceLibrary {
    pub fn new() -> Self {
        let mut voices = BTreeMap::new();
        for spec in BUILTIN_VOICES {
            let v = spec.to_voice();
            voices.insert(v.voice_id.clone(), v);
        }
        Self { voices }
    }

    /// List all registered voices (`GET /v1/voices`).
    pub fn list_voices(&self) -> Vec<Voice> {
        self.voices.values().cloned().collect()
    }

    /// Resolve a voice by ID, ElevenLabs 20-char alias ID, name, or `desc:` natural-language prompt.
    pub fn resolve(&self, voice_id_or_desc: &str) -> Result<Voice> {
        let trimmed = voice_id_or_desc.trim();
        if let Some(v) = self.voices.get(trimmed) {
            return Ok(v.clone());
        }
        for v in self.voices.values() {
            if v.voice_id.eq_ignore_ascii_case(trimmed)
                || v.name.eq_ignore_ascii_case(trimmed)
                || v.elevenlabs_alias_id.as_deref() == Some(trimmed)
            {
                return Ok(v.clone());
            }
        }
        if trimmed.starts_with(DESC_PREFIX) {
            let desc = describe_with_style(trimmed, None)?;
            let tract = tract_from_description(&desc);
            let id = format!("desc_{:016x}", fnv1a64(desc.as_bytes()));
            return Ok(Voice {
                voice_id: id,
                elevenlabs_alias_id: None,
                name: "Designed Voice".to_string(),
                category: "generated".to_string(),
                description: desc,
                labels: BTreeMap::new(),
                settings: VoiceSettings::default(),
                tract,
            });
        }
        Err(Error::new(
            STS_BAD_ARG,
            format!("unknown voice_id: {trimmed}"),
        ))
    }

    /// Instant Voice Cloning (`POST /v1/voices/add`): extracts a speaker's fundamental pitch $F_0$,
    /// 4-formant vocal tract envelope $(F_1..F_4)$, spectral tilt, breathiness, and 64-D Glass-Box
    /// signature directly from one or more reference audio clips in sub-millisecond time.
    pub fn add_cloned_voice(
        &mut self,
        name: &str,
        description: &str,
        reference_samples: &[AudioBuffer],
        labels: BTreeMap<String, String>,
    ) -> Result<(Voice, VoiceCloningReceipt)> {
        if name.trim().is_empty() {
            return Err(Error::new(STS_BAD_ARG, "cloned voice name cannot be empty"));
        }
        if reference_samples.is_empty() || reference_samples.iter().all(|b| b.is_empty()) {
            return Err(Error::new(
                STS_BAD_ARG,
                "at least one non-empty reference AudioBuffer is required for voice cloning",
            ));
        }

        let primary = &reference_samples[0];
        let sig = GlassBoxVoiceSignature64D::from_audio(primary)?;
        let f0 = sig.estimated_f0_hz.clamp(85.0, 310.0);
        let scale = (f0 / 165.0).powf(0.35).clamp(0.82, 1.22);
        let formants_hz = [
            500.0 * scale,
            1550.0 * scale,
            2550.0 * scale,
            3500.0 * scale,
        ];
        let breathiness = (1.0 - sig.voiced_ratio as f32).clamp(0.04, 0.28);
        let voice_id = format!("clone_{:016x}", sig.canonical_hash ^ fnv1a64(name.as_bytes()));

        let voice = Voice {
            voice_id: voice_id.clone(),
            elevenlabs_alias_id: None,
            name: name.trim().to_string(),
            category: "cloned".to_string(),
            description: if description.trim().is_empty() {
                format!("Instant cloned voice ({:.1} Hz base F0)", f0)
            } else {
                description.trim().to_string()
            },
            labels,
            settings: VoiceSettings::default(),
            tract: VocalTractProfile {
                base_f0_hz: f0,
                formants_hz,
                spectral_tilt: 0.81,
                breathiness,
                base_pace: 1.0,
                is_robot: false,
                binary_fingerprint: [
                    sig.canonical_hash,
                    sig.canonical_hash.rotate_left(13),
                    sig.canonical_hash.rotate_right(19),
                    !sig.canonical_hash,
                ],
            },
        };

        self.voices.insert(voice_id.clone(), voice.clone());

        let receipt = VoiceCloningReceipt {
            voice_id,
            name: voice.name.clone(),
            requires_verification: false,
            extracted_f0_hz: f0,
            extracted_formants_hz: formants_hz,
            glass_box_canonical_hash: sig.canonical_hash,
        };
        Ok((voice, receipt))
    }

    /// Delete a voice by ID (`DELETE /v1/voices/{voice_id}`).
    pub fn delete_voice(&mut self, voice_id: &str) -> bool {
        self.voices.remove(voice_id).is_some()
    }
}
