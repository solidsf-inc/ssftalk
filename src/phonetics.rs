//! Pure-Rust Grapheme-to-Phoneme (`g2p_english`), Arpabet/IPA Acoustic Targets,
//! ElevenLabs Pronunciation Dictionaries, and Inline Expression/Drama Tag Parsing (`phonetics`).

use serde::{Deserialize, Serialize};

/// Arpabet / IPA phoneme inventory with vocal-tract formant targets and intrinsic durations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phoneme {
    // Vowels
    AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW,
    // Voiced Consonants (Nasals, Liquids, Glides, Voiced Stops/Fricatives)
    B, D, DH, G, JH, L, M, N, NG, R, V, W, Y, Z, ZH,
    // Unvoiced Consonants (Plosives, Fricatives, Aspirates)
    CH, F, HH, K, P, S, SH, T, TH,
    // Paralinguistic & Silence
    LaughBurst,
    SighBreath,
    Sil,
}

impl Phoneme {
    /// Four formant frequencies `[F1, F2, F3, F4]` in Hz for this phoneme.
    #[inline]
    pub const fn formants_hz(&self) -> [f32; 4] {
        match self {
            Self::IY => [280.0, 2250.0, 2950.0, 3650.0],
            Self::IH => [400.0, 1950.0, 2600.0, 3550.0],
            Self::EY => [440.0, 2050.0, 2680.0, 3600.0],
            Self::EH => [530.0, 1820.0, 2520.0, 3500.0],
            Self::AE => [660.0, 1720.0, 2420.0, 3480.0],
            Self::AA => [730.0, 1090.0, 2440.0, 3400.0],
            Self::AO => [570.0, 840.0, 2410.0, 3350.0],
            Self::OW => [460.0, 980.0, 2380.0, 3350.0],
            Self::UH => [440.0, 1020.0, 2240.0, 3320.0],
            Self::UW => [300.0, 870.0, 2240.0, 3300.0],
            Self::AH => [520.0, 1190.0, 2390.0, 3420.0],
            Self::ER => [490.0, 1350.0, 1690.0, 3350.0],
            Self::AY => [680.0, 1450.0, 2480.0, 3480.0],
            Self::AW => [650.0, 1150.0, 2400.0, 3400.0],
            Self::OY => [520.0, 1380.0, 2450.0, 3450.0],
            Self::M | Self::N | Self::NG => [280.0, 1250.0, 2450.0, 3350.0],
            Self::L => [360.0, 1150.0, 2580.0, 3450.0],
            Self::R => [380.0, 1280.0, 1680.0, 3350.0],
            Self::W => [310.0, 780.0, 2220.0, 3300.0],
            Self::Y => [290.0, 2150.0, 2850.0, 3600.0],
            Self::S | Self::Z => [420.0, 1800.0, 3100.0, 4200.0],
            Self::SH | Self::ZH | Self::CH | Self::JH => [420.0, 1850.0, 2650.0, 3600.0],
            Self::F | Self::V | Self::TH | Self::DH => [450.0, 1450.0, 2550.0, 3550.0],
            Self::HH | Self::SighBreath => [500.0, 1500.0, 2500.0, 3500.0],
            Self::LaughBurst => [580.0, 1680.0, 2620.0, 3600.0],
            _ => [420.0, 1480.0, 2480.0, 3450.0],
        }
    }

    /// Intrinsic duration in 24 kHz acoustic frames (`1 frame = 256 samples = 10.67 ms`).
    #[inline]
    pub const fn intrinsic_frames(&self) -> usize {
        match self {
            Self::AA | Self::AE | Self::AO | Self::AW | Self::AY | Self::EY | Self::IY | Self::OW
            | Self::OY | Self::UW => 7,
            Self::EH | Self::ER | Self::IH | Self::UH | Self::AH => 5,
            Self::M | Self::N | Self::NG | Self::L | Self::R | Self::W | Self::Y => 4,
            Self::S | Self::SH | Self::Z | Self::ZH | Self::F | Self::TH => 5,
            Self::B | Self::D | Self::G | Self::P | Self::T | Self::K | Self::CH | Self::JH
            | Self::V | Self::DH | Self::HH => 3,
            Self::LaughBurst => 8,
            Self::SighBreath => 10,
            Self::Sil => 6,
        }
    }

    /// Harmonic voicing degree in `[0.0, 1.0]`.
    #[inline]
    pub const fn voicing(&self) -> f32 {
        match self {
            Self::AA | Self::AE | Self::AH | Self::AO | Self::AW | Self::AY | Self::EH
            | Self::ER | Self::EY | Self::IH | Self::IY | Self::OW | Self::OY | Self::UH
            | Self::UW => 0.96,
            Self::M | Self::N | Self::NG | Self::L | Self::R | Self::W | Self::Y => 0.88,
            Self::B | Self::D | Self::G | Self::V | Self::DH | Self::Z | Self::ZH | Self::JH => 0.68,
            Self::LaughBurst => 0.82,
            Self::SighBreath | Self::HH => 0.22,
            Self::CH | Self::F | Self::K | Self::P | Self::S | Self::SH | Self::T | Self::TH => 0.06,
            Self::Sil => 0.0,
        }
    }

    /// Aperiodic aspiration / frication breathiness in `[0.0, 1.0]`.
    #[inline]
    pub const fn breathiness(&self) -> f32 {
        match self {
            Self::S | Self::SH | Self::F | Self::TH | Self::CH => 0.88,
            Self::HH | Self::SighBreath => 0.92,
            Self::Z | Self::ZH | Self::V | Self::DH | Self::JH => 0.52,
            Self::P | Self::T | Self::K => 0.65,
            Self::LaughBurst => 0.45,
            Self::Sil => 0.0,
            _ => 0.08,
        }
    }

    /// Relative acoustic energy weight in `[0.0, 1.0]`.
    #[inline]
    pub const fn energy_weight(&self) -> f32 {
        match self {
            Self::AA | Self::AE | Self::AO | Self::AW | Self::AY | Self::EY | Self::OW
            | Self::OY => 0.95,
            Self::EH | Self::ER | Self::IH | Self::IY | Self::UH | Self::UW | Self::AH => 0.85,
            Self::LaughBurst => 0.92,
            Self::M | Self::N | Self::NG | Self::L | Self::R | Self::W | Self::Y => 0.65,
            Self::S | Self::SH | Self::Z | Self::ZH | Self::JH | Self::CH => 0.48,
            Self::SighBreath => 0.40,
            Self::Sil => 0.0,
            _ => 0.36,
        }
    }
}

/// Token emitted by `g2p_english`, linking a `Phoneme` to its source character index and prosodic stress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhonemeToken {
    pub phoneme: Phoneme,
    /// Lexical stress (`0` = unstressed, `1` = primary stress, `2` = secondary stress).
    pub stress: u8,
    /// Character index in the normalized input string for ElevenLabs `/with-timestamps` alignment.
    pub char_index: usize,
    /// Word index in the utterance.
    pub word_index: usize,
    /// True if this phoneme is the final phoneme of a word.
    pub word_boundary: bool,
}

/// Convert English text (with optional inline `<chuckle>`, `<laugh>`, `<sigh>`, `<whisper>` markers)
/// into a frame-ready sequence of `PhonemeToken`s with character alignment indices.
pub fn g2p_english(text: &str) -> Vec<PhonemeToken> {
    let mut tokens = Vec::with_capacity(text.len() + 4);
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let mut word_index = 0usize;

    while i < n {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        // Check inline expression markers like <laugh>, <chuckle>, <sigh>, <gasp>
        if ch == '<' {
            if let Some(rel_end) = chars[i..].iter().position(|&c| c == '>') {
                let tag: String = chars[i + 1..i + rel_end]
                    .iter()
                    .collect::<String>()
                    .to_ascii_lowercase();
                match tag.trim() {
                    "laugh" | "chuckle" | "giggle" => {
                        tokens.push(PhonemeToken {
                            phoneme: Phoneme::LaughBurst,
                            stress: 1,
                            char_index: i,
                            word_index,
                            word_boundary: false,
                        });
                        tokens.push(PhonemeToken {
                            phoneme: Phoneme::AH,
                            stress: 1,
                            char_index: i,
                            word_index,
                            word_boundary: true,
                        });
                    }
                    "sigh" | "gasp" | "breath" | "pant" => {
                        tokens.push(PhonemeToken {
                            phoneme: Phoneme::SighBreath,
                            stress: 0,
                            char_index: i,
                            word_index,
                            word_boundary: true,
                        });
                    }
                    _ => {}
                }
                i += rel_end + 1;
                continue;
            }
        }

        // Punctuation pauses
        if matches!(ch, '.' | '!' | '?' | ';' | ':' | ',' | '—') {
            tokens.push(PhonemeToken {
                phoneme: Phoneme::Sil,
                stress: 0,
                char_index: i,
                word_index,
                word_boundary: true,
            });
            i += 1;
            continue;
        }

        if !ch.is_alphabetic() && !ch.is_ascii_digit() {
            i += 1;
            continue;
        }

        // Collect contiguous word
        let w_start = i;
        while i < n && (chars[i].is_alphabetic() || chars[i].is_ascii_digit() || chars[i] == '\'') {
            i += 1;
        }
        let word_str: String = chars[w_start..i].iter().collect();
        let lower = word_str.to_ascii_lowercase();
        let word_phs = word_to_phonemes(&lower, w_start);
        let ph_len = word_phs.len();
        for (pi, (ph, stress, c_idx)) in word_phs.into_iter().enumerate() {
            tokens.push(PhonemeToken {
                phoneme: ph,
                stress,
                char_index: c_idx,
                word_index,
                word_boundary: pi + 1 == ph_len,
            });
        }
        word_index += 1;
    }

    if tokens.is_empty() {
        tokens.push(PhonemeToken {
            phoneme: Phoneme::Sil,
            stress: 0,
            char_index: 0,
            word_index: 0,
            word_boundary: true,
        });
    }

    tokens
}

fn word_to_phonemes(lower: &str, base_char_idx: usize) -> Vec<(Phoneme, u8, usize)> {
    // High-frequency English lexicon exceptions
    match lower {
        "the" => return vec![(Phoneme::DH, 0, base_char_idx), (Phoneme::AH, 0, base_char_idx + 1)],
        "a" => return vec![(Phoneme::AH, 0, base_char_idx)],
        "an" => return vec![(Phoneme::AE, 0, base_char_idx), (Phoneme::N, 0, base_char_idx + 1)],
        "and" => return vec![
            (Phoneme::AE, 0, base_char_idx),
            (Phoneme::N, 0, base_char_idx + 1),
            (Phoneme::D, 0, base_char_idx + 2),
        ],
        "to" | "too" | "two" => {
            return vec![(Phoneme::T, 0, base_char_idx), (Phoneme::UW, 1, base_char_idx + 1)]
        }
        "of" => return vec![(Phoneme::AH, 0, base_char_idx), (Phoneme::V, 0, base_char_idx + 1)],
        "in" => return vec![(Phoneme::IH, 0, base_char_idx), (Phoneme::N, 0, base_char_idx + 1)],
        "is" => return vec![(Phoneme::IH, 0, base_char_idx), (Phoneme::Z, 0, base_char_idx + 1)],
        "you" => return vec![(Phoneme::Y, 1, base_char_idx), (Phoneme::UW, 1, base_char_idx + 1)],
        "are" => return vec![(Phoneme::AA, 1, base_char_idx), (Phoneme::R, 0, base_char_idx + 1)],
        "hello" => {
            return vec![
                (Phoneme::HH, 0, base_char_idx),
                (Phoneme::EH, 0, base_char_idx + 1),
                (Phoneme::L, 1, base_char_idx + 2),
                (Phoneme::OW, 1, base_char_idx + 4),
            ]
        }
        "world" => {
            return vec![
                (Phoneme::W, 1, base_char_idx),
                (Phoneme::ER, 1, base_char_idx + 1),
                (Phoneme::L, 0, base_char_idx + 3),
                (Phoneme::D, 0, base_char_idx + 4),
            ]
        }
        "solidsf" => {
            return vec![
                (Phoneme::S, 1, base_char_idx),
                (Phoneme::AA, 1, base_char_idx + 1),
                (Phoneme::L, 0, base_char_idx + 2),
                (Phoneme::IH, 0, base_char_idx + 3),
                (Phoneme::D, 0, base_char_idx + 4),
                (Phoneme::EH, 1, base_char_idx + 5),
                (Phoneme::S, 0, base_char_idx + 5),
                (Phoneme::EH, 1, base_char_idx + 6),
                (Phoneme::F, 0, base_char_idx + 6),
            ]
        }
        _ => {}
    }

    let bytes = lower.as_bytes();
    let n = bytes.len();
    let mut out = Vec::with_capacity(n);
    let mut k = 0usize;
    let mut seen_primary_vowel = false;

    while k < n {
        let c_idx = base_char_idx + k;
        let b0 = bytes[k];
        let b1 = bytes.get(k + 1).copied().unwrap_or(0);

        // Digraph rules
        match (b0, b1) {
            (b't', b'h') => {
                out.push((Phoneme::TH, 0, c_idx));
                k += 2;
                continue;
            }
            (b's', b'h') => {
                out.push((Phoneme::SH, 0, c_idx));
                k += 2;
                continue;
            }
            (b'c', b'h') => {
                out.push((Phoneme::CH, 0, c_idx));
                k += 2;
                continue;
            }
            (b'p', b'h') => {
                out.push((Phoneme::F, 0, c_idx));
                k += 2;
                continue;
            }
            (b'n', b'g') => {
                out.push((Phoneme::NG, 0, c_idx));
                k += 2;
                continue;
            }
            (b'e', b'e') | (b'e', b'a') => {
                let st = if !seen_primary_vowel {
                    seen_primary_vowel = true;
                    1
                } else {
                    0
                };
                out.push((Phoneme::IY, st, c_idx));
                k += 2;
                continue;
            }
            (b'o', b'o') => {
                let st = if !seen_primary_vowel {
                    seen_primary_vowel = true;
                    1
                } else {
                    0
                };
                out.push((Phoneme::UW, st, c_idx));
                k += 2;
                continue;
            }
            (b'o', b'u') | (b'o', b'w') => {
                let st = if !seen_primary_vowel {
                    seen_primary_vowel = true;
                    1
                } else {
                    0
                };
                out.push((Phoneme::AW, st, c_idx));
                k += 2;
                continue;
            }
            (b'a', b'i') | (b'a', b'y') => {
                let st = if !seen_primary_vowel {
                    seen_primary_vowel = true;
                    1
                } else {
                    0
                };
                out.push((Phoneme::EY, st, c_idx));
                k += 2;
                continue;
            }
            (b'e', b'r') | (b'i', b'r') | (b'u', b'r') => {
                let st = if !seen_primary_vowel {
                    seen_primary_vowel = true;
                    1
                } else {
                    0
                };
                out.push((Phoneme::ER, st, c_idx));
                k += 2;
                continue;
            }
            _ => {}
        }

        // Trailing silent 'e'
        if b0 == b'e' && k + 1 == n && n > 2 && seen_primary_vowel {
            k += 1;
            continue;
        }

        let (ph, is_vowel) = match b0 {
            b'a' => (Phoneme::AE, true),
            b'e' => (Phoneme::EH, true),
            b'i' | b'y' => (Phoneme::IH, true),
            b'o' => (Phoneme::AA, true),
            b'u' => (Phoneme::AH, true),
            b'b' => (Phoneme::B, false),
            b'c' | b'k' | b'q' => (Phoneme::K, false),
            b'd' => (Phoneme::D, false),
            b'f' => (Phoneme::F, false),
            b'g' => (Phoneme::G, false),
            b'h' => (Phoneme::HH, false),
            b'j' => (Phoneme::JH, false),
            b'l' => (Phoneme::L, false),
            b'm' => (Phoneme::M, false),
            b'n' => (Phoneme::N, false),
            b'p' => (Phoneme::P, false),
            b'r' => (Phoneme::R, false),
            b's' | b'x' => (Phoneme::S, false),
            b't' => (Phoneme::T, false),
            b'v' => (Phoneme::V, false),
            b'w' => (Phoneme::W, false),
            b'z' => (Phoneme::Z, false),
            b'0'..=b'9' => (Phoneme::AH, true),
            _ => {
                k += 1;
                continue;
            }
        };

        let stress = if is_vowel && !seen_primary_vowel {
            seen_primary_vowel = true;
            1
        } else {
            0
        };
        out.push((ph, stress, c_idx));
        k += 1;
    }

    if out.is_empty() {
        out.push((Phoneme::AH, 0, base_char_idx));
    }
    out
}

/// ElevenLabs-compatible Pronunciation Dictionary entry (`alias` or `phoneme` rule).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationEntry {
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub case_sensitive: bool,
}

/// ElevenLabs-compatible `pronunciation_dictionary_locators` reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationDictionaryLocator {
    pub pronunciation_dictionary_id: String,
    pub version_id: Option<String>,
}

/// Pronunciation dictionary resource (`POST /v1/pronunciation-dictionaries/add-from-file`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PronunciationDictionary {
    pub id: String,
    pub version_id: String,
    pub name: String,
    pub rules: Vec<PronunciationEntry>,
}

/// Span of prepared speech or timed silence after parsing inline expression and break tags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Span {
    Speech { text: String },
    Silence { samples: usize },
}

/// Result of `prepare_input` after pronunciation replacement and inline tag normalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedInput {
    pub spans: Vec<Span>,
    pub inferred_style: Option<String>,
}

/// Apply pronunciation dictionary replacements on word boundaries.
pub fn apply_pronunciations(text: &str, entries: &[PronunciationEntry]) -> String {
    if entries.is_empty() || text.is_empty() {
        return text.to_owned();
    }
    let mut current = text.to_owned();
    for entry in entries {
        let key = entry.key.trim();
        let val = entry.value.trim();
        if key.is_empty() {
            continue;
        }
        current = replace_word_bounded(&current, key, val, entry.case_sensitive);
    }
    current
}

fn replace_word_bounded(haystack: &str, needle: &str, replacement: &str, case_sensitive: bool) -> String {
    let h_chars: Vec<char> = haystack.chars().collect();
    let n_chars: Vec<char> = needle.chars().collect();
    if n_chars.is_empty() || n_chars.len() > h_chars.len() {
        return haystack.to_owned();
    }
    let mut out = String::with_capacity(haystack.len());
    let mut i = 0usize;
    while i < h_chars.len() {
        if i + n_chars.len() <= h_chars.len() {
            let matches = h_chars[i..i + n_chars.len()]
                .iter()
                .zip(n_chars.iter())
                .all(|(&hc, &nc)| {
                    if case_sensitive {
                        hc == nc
                    } else {
                        hc.eq_ignore_ascii_case(&nc)
                    }
                });
            let prev_ok = i == 0 || !h_chars[i - 1].is_alphanumeric();
            let next_ok =
                i + n_chars.len() == h_chars.len() || !h_chars[i + n_chars.len()].is_alphanumeric();
            if matches && prev_ok && next_ok {
                out.push_str(replacement);
                i += n_chars.len();
                continue;
            }
        }
        out.push(h_chars[i]);
        i += 1;
    }
    out
}

/// Parse inline Fish/Maya/ElevenLabs expression tags (`[laugh]`, `[chuckle]`, `[short-break]`,
/// `<whisper>...</whisper>`) and pronunciation dictionary entries into `PreparedInput`.
pub fn prepare_input(
    raw: &str,
    pronunciations: &[PronunciationEntry],
    sample_rate: u32,
) -> PreparedInput {
    let text = apply_pronunciations(raw, pronunciations);
    let sr = sample_rate.max(8_000) as usize;
    let mut inferred_style: Option<String> = None;
    let mut spans: Vec<Span> = Vec::new();
    let mut cur_speech = String::new();

    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '[' || c == '<' {
            let close_ch = if c == '[' { ']' } else { '>' };
            if let Some(rel) = chars[i + 1..].iter().position(|&x| x == close_ch) {
                let raw_tag: String = chars[i + 1..i + 1 + rel].iter().collect();
                let tag = raw_tag.trim().trim_start_matches('/').to_ascii_lowercase();
                let is_closing = raw_tag.trim().starts_with('/');

                match tag.as_str() {
                    "short-break" | "short_break" => {
                        let trimmed = cur_speech.trim().to_string();
                        if !trimmed.is_empty() {
                            spans.push(Span::Speech { text: trimmed });
                            cur_speech.clear();
                        }
                        spans.push(Span::Silence {
                            samples: (sr * 180) / 1000,
                        });
                        i += rel + 2;
                        continue;
                    }
                    "break" | "pause" => {
                        let trimmed = cur_speech.trim().to_string();
                        if !trimmed.is_empty() {
                            spans.push(Span::Speech { text: trimmed });
                            cur_speech.clear();
                        }
                        spans.push(Span::Silence {
                            samples: (sr * 400) / 1000,
                        });
                        i += rel + 2;
                        continue;
                    }
                    "long-break" | "long_break" => {
                        let trimmed = cur_speech.trim().to_string();
                        if !trimmed.is_empty() {
                            spans.push(Span::Speech { text: trimmed });
                            cur_speech.clear();
                        }
                        spans.push(Span::Silence { samples: sr });
                        i += rel + 2;
                        continue;
                    }
                    "laugh" | "chuckle" | "giggle" | "sigh" | "gasp" => {
                        if !is_closing {
                            if !cur_speech.ends_with(' ') && !cur_speech.is_empty() {
                                cur_speech.push(' ');
                            }
                            cur_speech.push_str(&format!("<{tag}> "));
                        }
                        i += rel + 2;
                        continue;
                    }
                    "whisper" | "amused" | "excited" | "sad" | "angry" | "curious" | "serious"
                    | "warm" | "deadpan" | "empathetic" | "urgent" | "calm" => {
                        if !is_closing && inferred_style.is_none() {
                            inferred_style = Some(tag.clone());
                        }
                        if tag == "whisper" && !is_closing {
                            if !cur_speech.ends_with(' ') && !cur_speech.is_empty() {
                                cur_speech.push(' ');
                            }
                            cur_speech.push_str("<whisper> ");
                        }
                        i += rel + 2;
                        continue;
                    }
                    _ => {}
                }
            }
        }
        cur_speech.push(c);
        i += 1;
    }

    let trimmed = validate_whitespace(&cur_speech);
    if !trimmed.is_empty() {
        spans.push(Span::Speech { text: trimmed });
    }
    for span in &mut spans {
        if let Span::Speech { text } = span {
            *text = validate_whitespace(text);
        }
    }
    PreparedInput {
        spans,
        inferred_style,
    }
}

fn validate_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.trim().chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

/// Strip all inline expression/drama tags (`[laugh]`, `<whisper>`, `[break]`) leaving clean transcript prose.
pub fn strip_expression_tags(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '[' || c == '<' {
            let close_ch = if c == '[' { ']' } else { '>' };
            if let Some(rel) = chars[i + 1..].iter().position(|&x| x == close_ch) {
                i += rel + 2;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    validate_whitespace(&out)
}
