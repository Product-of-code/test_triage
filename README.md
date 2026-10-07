# test_triage
# AI Symptom Checker & Triage Tool

A simple NLP-powered healthcare application built for **ITAI 2372 – Module 03: AI in Healthcare**.
It takes symptoms typed in plain English, analyzes them with natural language processing, and returns
possible condition categories plus a suggested **triage level**: Emergency, Urgent care, Primary care, or Self-care.

> **Disclaimer:** Educational project only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.
> In an emergency, call 911.

## Chosen innovation
AI-powered symptom checkers / triage tools (similar in concept to commercial tools such as Ada Health or Babylon's
triage assistant).

### Research summary
- **Accuracy is limited.** A peer-reviewed study re-tested 22 symptom-checker apps on the same case vignettes used in a
  2015 study. Median triage accuracy was roughly 56% in 2020 versus roughly 59% in 2015, so it did not improve.
  The apps also became less cautious (fewer over-triage errors relative to under-triage errors) and still missed a large
  share of true emergencies. Few apps beat laypeople at deciding between emergency care and self-care, and accuracy varied
  widely from app to app. *(Schmieding et al., "Triage Accuracy of Symptom Checker Apps: 5-Year Follow-up Evaluation,"
  Journal of Medical Internet Research, 2022.)*
- **Real-world adoption.** Ada Health's symptom assessment and care-navigation tool has been deployed across Jefferson
  Health's network (18 hospitals, 50+ outpatient sites), letting patients decide whether to self-manage, book a visit,
  use telehealth, or chat with a physician. *(MedCity News, April 2023.)*
- **Design takeaways applied in this project:** because under-triage is the most dangerous failure, this tool uses
  hard-coded red-flag rules that always escalate to Emergency, picks the more urgent level on ties, and always shows a
  disclaimer.
  - **Real-world safety evidence is mixed.** In a 2024 prospective study at a Swiss hospital, an AI symptom checker
  (the SMASS Pathfinder, built on a transparent neural network) was used with 2,543 adult patients. Researchers compared
  its recommendations with expert panel reviews and found no strictly defined case of hazardous undertriage. Over-triage
  was about 18%. The authors concluded it appeared safe in that setting. This contrasts with the 2022 app-comparison study
  and shows results depend heavily on the specific tool. *(Meer et al., Journal of Medical Internet Research, 2024.)*
### Sources
- [Triage Accuracy of Symptom Checker Apps: 5-Year Follow-up Evaluation (JMIR, 2022)](https://doaj.org/article/ee5dee6d03d842c19d9b41472dbb7dc4)
- [Ada Health deploys its symptom assessment and care navigation tech across Jefferson Health (MedCity News, 2023)](https://medcitynews.com/2023/04/ada-health-deploys-its-symptom-assessment-care-navigation-tech-across-jefferson-health/)
- [A Symptom-Checker for Adult Patients Visiting an Interdisciplinary Emergency Care Center and the Safety of Patient Self-Triage (JMIR, 2024)](https://doaj.org/article/9695fe7126364552bcdd38e611c7b36d)

## How it works
1. **Tokenization** – split text into words.
2. **Negation handling** – "no fever" removes *fever* from the symptom list.
3. **Stop-word removal** – drop filler words ("I", "have", "a", ...).
4. **Lemmatization** – reduce words to a base form (*sneezing → sneeze*, *headaches → headache*).
5. **Keyword matching** – compare tokens to a small hand-built knowledge base of 9 conditions and rank by overlap.
6. **Red-flag rules** – phrases like "chest pain" or "trouble breathing" always escalate to **Emergency**.
7. **Cautious triage** – if top matches tie, the most urgent triage level among them is used.

## Files
| File | Purpose |
|------|---------|
| `triage.py` | NLP pipeline, knowledge base, triage logic (well commented) |
| `app.py` | Streamlit web interface |
| `test_triage.py` | Automated tests |
| `requirements.txt` | Dependencies |

## Run it
```bash
pip install -r requirements.txt
python test_triage.py          # run the tests
python triage.py               # command-line demo
streamlit run app.py           # web app
```
NLTK data (punkt, stopwords, wordnet) downloads automatically on first run; if offline, simple built-in fallbacks are used.

## Ethical considerations
- **Data quality:** The knowledge base is tiny and hand-written; real tools need clinically validated data.
- **Bias:** Symptom presentation differs by age, sex, and ethnicity; a small dataset can't represent that, and
  real systems must be tested across demographic groups.
- **Privacy:** This app processes text in memory only and stores nothing. A production tool collecting health data
  would need encryption, consent, and HIPAA compliance.
- **Safety:** Red-flag rules fail safe toward higher urgency, and a disclaimer is always shown.
- **Limitations:** Keyword matching can't understand context, severity, duration, age, or medical history.

## Future improvements
Add severity/duration questions, fuzzy matching or a scikit-learn model (e.g., Naive Bayes) trained on a real dataset,
or an LLM with a safety-checked prompt, and evaluate with accuracy, precision, and recall on a held-out test set.

## AI assistance
Designed and written with help from Claude (Anthropic), then reviewed and tested by me.
