"""
app.py - Streamlit web interface for the AI Symptom Checker & Triage Tool.

Run with:  streamlit run app.py
"""

import streamlit as st

from triage import DISCLAIMER, analyze_symptoms

# Colors/icons for each triage level so the result is easy to scan.
LEVEL_STYLE = {
    "Emergency": ("🚨", st.error),
    "Urgent care": ("⚠️", st.warning),
    "Primary care": ("🩺", st.info),
    "Self-care": ("🏠", st.success),
}

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺")
st.title("🩺 AI Symptom Checker & Triage Tool")
st.caption("NLP-based demo project for ITAI 2372 - Module 03: AI in Healthcare")
st.warning(DISCLAIMER)

# Privacy by design: the text is analyzed in memory only and never saved or logged.
symptoms = st.text_area(
    "Describe your symptoms in your own words:",
    placeholder="e.g. I have a sore throat, a fever and swollen glands since yesterday",
    height=120,
)

if st.button("Analyze", type="primary"):
    if not symptoms.strip():
        st.write("Please enter your symptoms.")
    else:
        result = analyze_symptoms(symptoms)
        icon, show = LEVEL_STYLE[result.triage]
        show(f"{icon} **Triage level: {result.triage}**  \n{result.summary}")

        for reason in result.red_flags:
            st.error(f"Red flag: {reason}")

        if result.matches:
            st.subheader("Possible matches")
            for m in result.matches:
                with st.expander(f"{m.condition} - confidence: {m.confidence}"):
                    st.write(f"**Matched symptoms:** {', '.join(m.matched_keywords)}")
                    st.write(f"**Suggested care level:** {m.triage}")
                    st.write(m.advice)

        with st.expander("How the NLP worked (processed tokens)"):
            st.write(result.tokens)

st.markdown("---")
st.markdown(f"*{DISCLAIMER}*")
