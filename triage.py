"""
triage.py - core logic for the AI Symptom Checker & Triage Tool.

Pipeline (classic NLP, rule-based, fully explainable):
    raw text -> lowercase/tokenize -> negation handling -> stop-word removal
             -> lemmatization -> keyword matching against a knowledge base
             -> ranked possible conditions + triage level

Red-flag phrases (e.g. "chest pain", "can't breathe") are checked FIRST on the
raw text and always escalate to Emergency, no matter what else matches.

EDUCATIONAL PROJECT ONLY - not medical advice.
"""

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# 1. NLP helpers (NLTK if available, simple fallbacks otherwise)
# --------------------------------------------------------------------------

# Fallback stop words used if the NLTK corpus can't be downloaded.
_FALLBACK_STOPWORDS = {
    "i", "me", "my", "have", "has", "had", "am", "is", "are", "was", "were",
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "very", "really", "some", "also", "it", "its", "been", "feel", "feeling",
    "get", "getting", "got", "since", "about", "bad", "lot", "little", "bit",
}
# Words that flip the meaning of the NEXT word ("no fever", "without cough").
NEGATIONS = {"no", "not", "without", "never", "denies", "don't", "dont", "cannot_x"}

_nltk_ok = False
_lemmatizer = None
_stop_words = set(_FALLBACK_STOPWORDS)

try:  # Try to use NLTK, downloading small data packages once.
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        nltk.download(pkg, quiet=True)
    _stop_words = set(stopwords.words("english")) - NEGATIONS  # keep negations!
    _stop_words |= {"feel", "feeling", "get", "getting", "got", "bad", "lot", "bit"}
    _lemmatizer = WordNetLemmatizer()
    word_tokenize("test sentence")  # raises LookupError if punkt is missing
    _nltk_ok = True
except Exception:  # offline / NLTK missing -> fall back gracefully
    _nltk_ok = False


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    text = text.lower().replace("can't", "cannot").replace("cant ", "cannot ")
    if _nltk_ok:
        return [t for t in word_tokenize(text) if re.match(r"[a-z']+$", t)]
    return re.findall(r"[a-z']+", text)


def lemmatize(word: str) -> str:
    """Reduce a word to its dictionary form (e.g. 'sneezing' -> 'sneeze')."""
    if _nltk_ok and _lemmatizer:
        w = _lemmatizer.lemmatize(word, pos="v")      # try as verb first
        return _lemmatizer.lemmatize(w, pos="n")      # then as noun (headaches -> headache)
    # Crude suffix stripping fallback.
    for suffix in ("ing", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def preprocess(text: str) -> list[str]:
    """
    Turn raw symptom text into clean symptom tokens.
    Negated words are dropped: "no fever, sore throat" -> ['sore', 'throat'].
    """
    tokens = tokenize(text)
    cleaned, negate_next = [], 0
    for tok in tokens:
        if tok in NEGATIONS:
            negate_next = 2          # a negation covers the next couple of words
            continue
        if negate_next:
            negate_next -= 1
            if tok not in _stop_words:
                continue             # skip the negated symptom word
        if tok in _stop_words:
            continue
        cleaned.append(lemmatize(tok))
    return cleaned


# --------------------------------------------------------------------------
# 2. Knowledge base (small, hand-built dataset - see README for limitations)
# --------------------------------------------------------------------------

TRIAGE_ORDER = ["Self-care", "Primary care", "Urgent care", "Emergency"]

# Each condition: symptom keywords (already lemmatized), triage level, advice.
KNOWLEDGE_BASE = {
    "Common cold": {
        "keywords": {"cough", "sneeze", "runny", "nose", "congestion", "sore", "throat", "stuffy"},
        "triage": "Self-care",
        "advice": "Rest, fluids and over-the-counter relief usually help. See a doctor if it lasts over 10 days.",
    },
    "Influenza (flu)": {
        "keywords": {"fever", "chill", "ache", "body", "fatigue", "cough", "headache", "tired", "weak"},
        "triage": "Primary care",
        "advice": "Rest and fluids. Contact a doctor soon - antivirals work best early, especially for high-risk people.",
    },
    "Seasonal allergies": {
        "keywords": {"sneeze", "itchy", "eye", "runny", "nose", "watery", "congestion"},
        "triage": "Self-care",
        "advice": "Avoid triggers; antihistamines may help. See a doctor if symptoms are frequent or severe.",
    },
    "Strep throat": {
        "keywords": {"sore", "throat", "fever", "swollen", "gland", "swallow", "pain", "headache"},
        "triage": "Urgent care",
        "advice": "Needs a throat swab test; strep is treated with antibiotics.",
    },
    "Migraine": {
        "keywords": {"headache", "throb", "nausea", "light", "sensitivity", "vomit", "aura", "migraine"},
        "triage": "Primary care",
        "advice": "Rest in a dark, quiet room. See a doctor if migraines are new, frequent, or worsening.",
    },
    "Gastroenteritis (stomach bug)": {
        "keywords": {"vomit", "diarrhea", "nausea", "stomach", "cramp", "fever", "abdominal"},
        "triage": "Self-care",
        "advice": "Sip fluids to avoid dehydration. Seek care if you can't keep liquids down or it lasts over 2 days.",
    },
    "Urinary tract infection": {
        "keywords": {"burn", "urinate", "frequent", "urination", "pelvic", "pain", "urgency", "cloudy"},
        "triage": "Urgent care",
        "advice": "UTIs generally need antibiotics; get tested promptly, especially if you have fever or back pain.",
    },
    "Pneumonia (possible)": {
        "keywords": {"cough", "fever", "chest", "shortness", "breath", "phlegm", "chill", "fatigue"},
        "triage": "Urgent care",
        "advice": "Cough with fever and breathing trouble needs an in-person exam and possibly a chest X-ray.",
    },
    "Muscle strain / sprain": {
        "keywords": {"muscle", "pain", "swelling", "sprain", "strain", "ankle", "back", "sore", "bruise"},
        "triage": "Self-care",
        "advice": "Rest, ice, compression and elevation. See a doctor if you can't bear weight or swelling is severe.",
    },
}

# Phrases that ALWAYS trigger Emergency (checked on the raw lowercase text).
RED_FLAGS = {
    "chest pain": "Chest pain can signal a heart attack.",
    "chest pressure": "Chest pressure can signal a heart attack.",
    "cannot breathe": "Severe trouble breathing is an emergency.",
    "trouble breathing": "Severe trouble breathing is an emergency.",
    "difficulty breathing": "Severe trouble breathing is an emergency.",
    "shortness of breath": "Sudden shortness of breath can be an emergency.",
    "face drooping": "Face drooping is a possible stroke sign.",
    "slurred speech": "Slurred speech is a possible stroke sign.",
    "weakness on one side": "One-sided weakness is a possible stroke sign.",
    "severe bleeding": "Severe bleeding is an emergency.",
    "unconscious": "Loss of consciousness is an emergency.",
    "passed out": "Fainting with other symptoms can be an emergency.",
    "seizure": "A seizure needs emergency evaluation.",
    "suicidal": "If you are thinking about harming yourself, call or text 988 (Suicide & Crisis Lifeline) now.",
    "want to die": "If you are thinking about harming yourself, call or text 988 (Suicide & Crisis Lifeline) now.",
    "worst headache of my life": "A sudden, extremely severe headache can be an emergency.",
}

DISCLAIMER = (
    "This tool is for educational and informational purposes only and is NOT a "
    "substitute for professional medical advice, diagnosis, or treatment. "
    "If you think you may have a medical emergency, call 911."
)


# --------------------------------------------------------------------------
# 3. Matching and triage
# --------------------------------------------------------------------------

@dataclass
class Match:
    condition: str
    matched_keywords: list
    score: float          # 0-1, fraction of the condition's keywords found
    confidence: str       # Low / Medium / High (based on how many keywords matched)
    triage: str
    advice: str


@dataclass
class Result:
    triage: str
    summary: str
    red_flags: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    tokens: list = field(default_factory=list)


def _confidence(n_matched: int) -> str:
    return "High" if n_matched >= 4 else "Medium" if n_matched >= 3 else "Low"


def analyze_symptoms(text: str, top_n: int = 3) -> Result:
    """Main entry point: free-text symptoms in, ranked conditions + triage out."""
    raw = text.lower().replace("can't", "cannot")
    tokens = preprocess(text)

    # Step 1: red flags override everything else.
    flags = [reason for phrase, reason in RED_FLAGS.items() if phrase in raw]

    # Step 2: score each condition by keyword overlap.
    matches = []
    token_set = set(tokens)
    for name, info in KNOWLEDGE_BASE.items():
        hit = sorted(token_set & info["keywords"])
        if not hit:
            continue
        matches.append(Match(
            condition=name,
            matched_keywords=hit,
            score=len(hit) / len(info["keywords"]),
            confidence=_confidence(len(hit)),
            triage=info["triage"],
            advice=info["advice"],
        ))
    # Rank by number of matches first, then by coverage of that condition.
    matches.sort(key=lambda m: (len(m.matched_keywords), m.score), reverse=True)
    matches = matches[:top_n]

    # Step 3: choose the triage level.
    if flags:
        return Result("Emergency",
                      "Possible emergency. Call 911 or go to the nearest emergency room now.",
                      flags, matches, tokens)
    if not matches:
        return Result("Primary care",
                      "I couldn't match these symptoms to anything in my small knowledge base. "
                      "Please talk to a doctor or pharmacist, or add more detail.",
                      [], [], tokens)
    # Be cautious: use the MOST urgent level among the top matches that share the best hit count.
    best_hits = len(matches[0].matched_keywords)
    candidates = [m for m in matches if len(m.matched_keywords) == best_hits]
    triage = max((m.triage for m in candidates), key=TRIAGE_ORDER.index)
    return Result(triage, f"Most likely category: {matches[0].condition}.", [], matches, tokens)


if __name__ == "__main__":  # quick command-line demo
    print(DISCLAIMER, "\n")
    user = input("Describe your symptoms: ")
    res = analyze_symptoms(user)
    print(f"\nTriage level: {res.triage}\n{res.summary}")
    for f in res.red_flags:
        print("  RED FLAG:", f)
    for m in res.matches:
        print(f"  - {m.condition} ({m.confidence}, matched: {', '.join(m.matched_keywords)}) -> {m.advice}")
