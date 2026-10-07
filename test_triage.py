"""Simple tests for the triage logic. Run with:  python test_triage.py  (or pytest)"""

from triage import analyze_symptoms, preprocess


def test_preprocess_removes_stopwords_and_lemmatizes():
    tokens = preprocess("I have a severe headache and sneezing")
    assert "headache" in tokens
    assert "i" not in tokens and "a" not in tokens


def test_negation_drops_symptom():
    tokens = preprocess("sore throat but no fever")
    assert "fever" not in tokens
    assert "throat" in tokens


def test_red_flag_is_emergency():
    r = analyze_symptoms("I have chest pain and a mild cough")
    assert r.triage == "Emergency" and r.red_flags


def test_cold_is_self_care():
    r = analyze_symptoms("runny nose, sneezing and a stuffy nose")
    assert r.triage == "Self-care"
    assert r.matches[0].condition in ("Common cold", "Seasonal allergies")


def test_strep_is_urgent():
    r = analyze_symptoms("sore throat, fever, swollen glands, hard to swallow")
    assert r.triage == "Urgent care"
    assert r.matches[0].condition == "Strep throat"


def test_unknown_symptoms_fall_back_to_doctor():
    r = analyze_symptoms("my toenail looks purple")
    assert r.triage == "Primary care" and not r.matches


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
