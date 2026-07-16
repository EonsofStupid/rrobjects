//! Minimal, dependency-free text utilities shared across components.
//!
//! Deliberately simple and stable: the DevPULSE models do their own
//! tokenization; this exists so the default (weightless) components — the
//! deterministic embedder, the lexical reranker, the heuristic classifier —
//! agree on what a "token" is.

/// Lowercase the input and split it into alphanumeric tokens.
///
/// Runs of non-alphanumeric characters are separators. Empty tokens are
/// dropped. This is intentionally naive; it is a floor, not the model.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// A small English stopword set for the lexical fallbacks: articles,
/// conjunctions, prepositions, pronouns, and the question/command fillers that
/// pad natural-language queries ("how **do** I", "**tell** me **about**").
pub const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "what", "how", "why", "when", "where", "who",
    // query/command fillers and pronouns
    "do", "does", "did", "i", "me", "my", "we", "our", "you", "your", "can", "could", "would",
    "should", "please", "tell", "about", "give", "show", "need", "want", "get", "got", "from",
    "than", "also", "just", "am",
];

/// Whether `token` is a stopword.
pub fn is_stopword(token: &str) -> bool {
    STOPWORDS.contains(&token)
}

/// Tokenize and drop stopwords — the "content" tokens of a string.
pub fn content_tokens(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| !is_stopword(t))
        .collect()
}

// ---- analyzers ---------------------------------------------------------------
//
// A configurable pipeline: tokenizer × lowercase × stopwords × stemmer.
// The estate persists its analyzer at creation so the BM25 index and every
// later query agree on what a token is — an analyzer is part of the index's
// identity, not a per-call option.

use serde::{Deserialize, Serialize};

/// How raw text becomes candidate tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Tokenizer {
    /// Split on any non-alphanumeric rune (the classic word tokenizer).
    #[default]
    Word,
    /// Split on Unicode whitespace only (punctuation stays attached).
    Whitespace,
    /// Word tokens expanded into edge prefixes of `min..=max` chars —
    /// the autocomplete tokenizer ("connectome" → "con", "conn", …).
    Prefix {
        /// Shortest emitted prefix.
        min: usize,
        /// Longest emitted prefix (full token always included).
        max: usize,
    },
}

/// A text-analysis pipeline (tokenize → lowercase → stopwords → stem).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Analyzer {
    /// Tokenization strategy.
    #[serde(default)]
    pub tokenizer: Tokenizer,
    /// Lowercase tokens before filtering.
    #[serde(default = "yes")]
    pub lowercase: bool,
    /// Drop [`STOPWORDS`].
    #[serde(default = "yes")]
    pub stopwords: bool,
    /// Apply the Porter stemmer to each surviving token.
    #[serde(default)]
    pub stem: bool,
}

fn yes() -> bool {
    true
}

impl Default for Analyzer {
    /// The legacy pipeline every existing estate was built with:
    /// word tokens, lowercased, stopword-filtered, unstemmed.
    fn default() -> Self {
        Analyzer {
            tokenizer: Tokenizer::Word,
            lowercase: true,
            stopwords: true,
            stem: false,
        }
    }
}

impl Analyzer {
    /// The default pipeline with stemming on (English).
    pub fn stemming() -> Self {
        Analyzer {
            stem: true,
            ..Analyzer::default()
        }
    }

    /// An autocomplete pipeline: edge prefixes of `min..=max`, no stemming
    /// (stems would fight prefixes), stopwords kept out.
    pub fn autocomplete(min: usize, max: usize) -> Self {
        Analyzer {
            tokenizer: Tokenizer::Prefix { min, max },
            lowercase: true,
            stopwords: true,
            stem: false,
        }
    }

    /// Run the full pipeline.
    pub fn analyze(&self, text: &str) -> Vec<String> {
        let raw: Vec<String> = match &self.tokenizer {
            Tokenizer::Word => text
                .split(|c: char| !c.is_alphanumeric())
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect(),
            Tokenizer::Whitespace => text.split_whitespace().map(str::to_string).collect(),
            Tokenizer::Prefix { min, max } => {
                let (min, max) = ((*min).max(1), (*max).max(*min));
                let mut out = Vec::new();
                for tok in text.split(|c: char| !c.is_alphanumeric()) {
                    if tok.is_empty() {
                        continue;
                    }
                    let chars: Vec<char> = tok.chars().collect();
                    for n in min..=max.min(chars.len()) {
                        out.push(chars[..n].iter().collect());
                    }
                    if chars.len() > max {
                        out.push(tok.to_string());
                    }
                }
                out
            }
        };
        let mut out = Vec::with_capacity(raw.len());
        for mut t in raw {
            if self.lowercase {
                t = t.to_lowercase();
            }
            if self.stopwords && is_stopword(&t) {
                continue;
            }
            if self.stem {
                t = porter_stem(&t);
            }
            if !t.is_empty() {
                out.push(t);
            }
        }
        out
    }
}

// ---- Porter stemmer ----------------------------------------------------------
//
// Authored from M.F. Porter's published 1980 algorithm ("An algorithm for
// suffix stripping"): five rule steps over a consonant/vowel measure.
// ASCII-only by design — non-ASCII tokens pass through untouched.

/// Is `w[i]` a consonant under Porter's definition? (`y` is a consonant
/// only when it is not preceded by a consonant.)
fn is_cons(w: &[u8], i: usize) -> bool {
    match w[i] {
        b'a' | b'e' | b'i' | b'o' | b'u' => false,
        b'y' => i == 0 || !is_cons(w, i - 1),
        _ => true,
    }
}

/// Porter's measure m of `w[..len]`: the number of vowel→consonant
/// transitions ([C](VC)^m[V]).
fn measure(w: &[u8], len: usize) -> usize {
    let mut m = 0;
    let mut prev_vowel = false;
    for i in 0..len {
        let cons = is_cons(w, i);
        if cons && prev_vowel {
            m += 1;
        }
        prev_vowel = !cons;
    }
    m
}

/// *v*: does `w[..len]` contain a vowel?
fn has_vowel(w: &[u8], len: usize) -> bool {
    (0..len).any(|i| !is_cons(w, i))
}

/// *d: does `w[..len]` end with a double consonant?
fn ends_double_cons(w: &[u8], len: usize) -> bool {
    len >= 2 && w[len - 1] == w[len - 2] && is_cons(w, len - 1)
}

/// *o: does `w[..len]` end consonant-vowel-consonant, where the final
/// consonant is not w, x, or y?
fn ends_cvc(w: &[u8], len: usize) -> bool {
    len >= 3
        && is_cons(w, len - 3)
        && !is_cons(w, len - 2)
        && is_cons(w, len - 1)
        && !matches!(w[len - 1], b'w' | b'x' | b'y')
}

/// If `w[..len]` ends with `suffix`, the stem length before it; else None.
fn stem_before(w: &[u8], len: usize, suffix: &str) -> Option<usize> {
    let s = suffix.as_bytes();
    if len >= s.len() && &w[len - s.len()..len] == s {
        Some(len - s.len())
    } else {
        None
    }
}

/// Replace the suffix at `stem` with `to`, returning the new length.
fn set_suffix(w: &mut Vec<u8>, stem: usize, to: &str) -> usize {
    w.truncate(stem);
    w.extend_from_slice(to.as_bytes());
    w.len()
}

/// Try a (from → to) rule table gated on `min_measure`; applies the first
/// suffix that matches (longest first is the caller's responsibility).
fn rule_table(w: &mut Vec<u8>, len: usize, min_measure: usize, rules: &[(&str, &str)]) -> usize {
    for (from, to) in rules {
        if let Some(stem) = stem_before(w, len, from) {
            if measure(w, stem) > min_measure {
                return set_suffix(w, stem, to);
            }
            return len; // matched but condition failed: step ends
        }
    }
    len
}

/// Stem one lowercase token with the Porter algorithm. Tokens shorter than
/// three characters, or containing non-ASCII-alphabetic bytes, pass through.
pub fn porter_stem(token: &str) -> String {
    if token.len() <= 2 || !token.bytes().all(|b| b.is_ascii_lowercase()) {
        return token.to_string();
    }
    let mut w: Vec<u8> = token.as_bytes().to_vec();
    let mut len = w.len();

    // Step 1a: plurals.
    if let Some(stem) = stem_before(&w, len, "sses") {
        len = set_suffix(&mut w, stem, "ss");
    } else if let Some(stem) = stem_before(&w, len, "ies") {
        len = set_suffix(&mut w, stem, "i");
    } else if stem_before(&w, len, "ss").is_some() {
        // keep
    } else if let Some(stem) = stem_before(&w, len, "s") {
        len = set_suffix(&mut w, stem, "");
    }

    // Step 1b: -eed / -ed / -ing.
    let mut cleanup_1b = false;
    if let Some(stem) = stem_before(&w, len, "eed") {
        if measure(&w, stem) > 0 {
            len = set_suffix(&mut w, stem, "ee");
        }
    } else if let Some(stem) = stem_before(&w, len, "ed") {
        if has_vowel(&w, stem) {
            len = set_suffix(&mut w, stem, "");
            cleanup_1b = true;
        }
    } else if let Some(stem) = stem_before(&w, len, "ing") {
        if has_vowel(&w, stem) {
            len = set_suffix(&mut w, stem, "");
            cleanup_1b = true;
        }
    }
    if cleanup_1b {
        if stem_before(&w, len, "at").is_some()
            || stem_before(&w, len, "bl").is_some()
            || stem_before(&w, len, "iz").is_some()
        {
            len = set_suffix(&mut w, len, "e");
        } else if ends_double_cons(&w, len) && !matches!(w[len - 1], b'l' | b's' | b'z') {
            len -= 1;
            w.truncate(len);
        } else if measure(&w, len) == 1 && ends_cvc(&w, len) {
            len = set_suffix(&mut w, len, "e");
        }
    }

    // Step 1c: y → i after a vowel.
    if let Some(stem) = stem_before(&w, len, "y") {
        if has_vowel(&w, stem) {
            len = set_suffix(&mut w, stem, "i");
        }
    }

    // Step 2 (m > 0): double suffixes, longest candidates first per ending.
    len = rule_table(
        &mut w,
        len,
        0,
        &[
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ],
    );

    // Step 3 (m > 0).
    len = rule_table(
        &mut w,
        len,
        0,
        &[
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ],
    );

    // Step 4 (m > 1): strip residual suffixes. -ion needs s/t before it.
    let step4: &[&str] = &[
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ou",
        "ism", "ate", "iti", "ous", "ive", "ize",
    ];
    let mut done4 = false;
    // Longest match first across the whole table.
    let mut best: Option<usize> = None;
    for from in step4 {
        if let Some(stem) = stem_before(&w, len, from) {
            if best.is_none_or(|b| stem < b) {
                best = Some(stem);
            }
        }
    }
    if let Some(stem) = stem_before(&w, len, "ion") {
        if stem >= 1 && matches!(w[stem - 1], b's' | b't') && best.is_none_or(|b| stem < b) {
            best = Some(stem);
        }
    }
    if let Some(stem) = best {
        if measure(&w, stem) > 1 {
            len = set_suffix(&mut w, stem, "");
            done4 = true;
        }
    }
    let _ = done4;

    // Step 5a: final -e.
    if let Some(stem) = stem_before(&w, len, "e") {
        let m = measure(&w, stem);
        if m > 1 || (m == 1 && !ends_cvc(&w, stem)) {
            len = set_suffix(&mut w, stem, "");
        }
    }
    // Step 5b: -ll → -l when m > 1.
    if ends_double_cons(&w, len) && w[len - 1] == b'l' && measure(&w, len) > 1 {
        len -= 1;
        w.truncate(len);
    }

    w.truncate(len);
    String::from_utf8(w).unwrap_or_else(|_| token.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical pairs from the published algorithm's own examples.
    #[test]
    fn porter_spec_pairs() {
        for (from, to) in [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caress", "caress"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("hopping", "hop"),
            ("falling", "fall"),
            ("filing", "file"),
            ("happy", "happi"),
            ("sky", "sky"),
            ("relational", "relat"),
            ("conditional", "condit"),
            ("rational", "ration"),
            ("digitizer", "digit"),
            ("operator", "oper"),
            ("vietnamization", "vietnam"),
            ("feudalism", "feudal"),
            ("hopefulness", "hope"),
            ("formaliti", "formal"),
            ("triplicate", "triplic"),
            ("formative", "form"),
            ("formalize", "formal"),
            ("electrical", "electr"),
            ("hopeful", "hope"),
            ("goodness", "good"),
            ("revival", "reviv"),
            ("allowance", "allow"),
            ("inference", "infer"),
            ("airliner", "airlin"),
            ("adjustable", "adjust"),
            ("replacement", "replac"),
            ("adjustment", "adjust"),
            ("dependent", "depend"),
            ("adoption", "adopt"),
            ("communism", "commun"),
            ("activate", "activ"),
            ("effective", "effect"),
            ("rate", "rate"),
            ("controlling", "control"),
            ("running", "run"),
        ] {
            assert_eq!(porter_stem(from), to, "{from} should stem to {to}");
        }
    }

    #[test]
    fn analyzer_pipelines() {
        // Legacy default: word + lowercase + stopwords, no stem.
        let legacy = Analyzer::default();
        assert_eq!(
            legacy.analyze("The Running Foxes!"),
            vec!["running", "foxes"]
        );
        // Same pipeline as content_tokens (index/query agreement contract).
        assert_eq!(
            legacy.analyze("How do I tune the estate?"),
            content_tokens("How do I tune the estate?")
        );

        // Stemming: inflections collapse.
        let stemming = Analyzer::stemming();
        assert_eq!(stemming.analyze("The Running Foxes!"), vec!["run", "fox"]);
        assert_eq!(
            stemming.analyze("connected connection connections"),
            vec!["connect", "connect", "connect"]
        );

        // Autocomplete: edge prefixes, full token kept beyond max.
        let auto = Analyzer::autocomplete(3, 5);
        let toks = auto.analyze("Connectome");
        assert_eq!(toks, vec!["con", "conn", "conne", "connectome"]);

        // Whitespace keeps punctuation attached.
        let ws = Analyzer {
            tokenizer: Tokenizer::Whitespace,
            lowercase: true,
            stopwords: false,
            stem: false,
        };
        assert_eq!(
            ws.analyze("keep-this together!"),
            vec!["keep-this", "together!"]
        );

        // Serde roundtrip + legacy default from empty JSON.
        let json = serde_json::to_string(&stemming).unwrap();
        assert_eq!(serde_json::from_str::<Analyzer>(&json).unwrap(), stemming);
        let empty: Analyzer = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, Analyzer::default());
    }
}
