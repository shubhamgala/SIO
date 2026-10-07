# SIO — Smart Indian Banking Voice Assistant

> A **voice-first banking assistant** designed for elderly users in India.  
> Speak in **Hindi, Hinglish, or English** — SIO understands and responds.

---

## 🗂️ Project Structure

```
SIO/
├── STT/
│   └── whisper_stt.py          # Module 1 — Speech to Text (OpenAI Whisper)
├── NLP/
│   └── intent_recognition.py   # Module 2 — Intent & Entity Recognition
├── Backend/
│   └── banking_logic.py        # Module 3 — Banking Actions & Dialogue (coming soon)
├── Frontend/                   # Module 5 — UI (coming soon)
├── main.py                     # Integration bridge: STT → NLP → Backend
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ✅ Modules Completed

### Module 1 — Speech to Text (`STT/whisper_stt.py`)
- Uses **OpenAI Whisper `base` model** (auto-downloaded on first run)
- Records **5 seconds** with a 3-second countdown (designed for elderly users)
- Supports **Hindi, Hinglish, and English** — auto-detects language
- Falls back to Hindi if a non-supported language is detected
- Post-processes Whisper output to remove hallucinations (leading fillers, bad currency formatting, trailing noise)
- Optimized for Realtek stereo mic arrays

### Module 2 — Intent & Entity Recognition (`NLP/intent_recognition.py`)
- Classifies user queries into **7 banking intents**:
  - Balance Enquiry, Fund Transfer, Transaction History
  - Mobile Recharge, Bill Payment, ATM Locator, Help
- Extracts **entities**: `amount` (monetary value) and `recipient` (name)
- Multilingual keyword dictionary (Hindi Devanagari + Hinglish + English)
- **5 entity extraction patterns** covering all common speech structures
- Architecture ready for **IndicBERT fine-tuning** — set `INDICBERT_LOCAL_PATH` in `intent_recognition.py` when weights are ready

---

## 🚀 Setup (for teammates)

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/SIO.git
cd SIO
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate it
- **Windows:**  `.\venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** On Windows, if `pip` is blocked by policy, use:  
> `python -m pip install -r requirements.txt`

### 5. Run the full pipeline
```bash
python main.py
```
Speak when prompted. The pipeline will transcribe your speech and classify the banking intent.

---

## 🧪 Running Module Tests Individually

```bash
# Test only Speech-to-Text (Module 1)
python STT/whisper_stt.py

# Test only Intent Recognition (Module 2)
python NLP/intent_recognition.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Speech to Text | [OpenAI Whisper](https://github.com/openai/whisper) `base` model |
| Intent Recognition | Keyword + Pattern Matching (IndicBERT-ready) |
| Audio Recording | `sounddevice` + `scipy` |
| NLP Framework | `transformers` (HuggingFace) |
| Language | Python 3.10+ |

---

## 📋 Supported Languages

| Language | Example |
|---|---|
| English | "Send Raju 5000 rupees" |
| Hinglish | "Raju ko 5000 bhejo" |
| Hindi | "रमेश को 500 रुपये ट्रांसफर कर दो" |

---

## 🗺️ Roadmap

- [x] Module 1 — Speech to Text
- [x] Module 2 — Intent & Entity Recognition
- [ ] Module 3 — Banking Logic & Voice Confirmation Dialogue
- [ ] Module 4 — Text to Speech Response
- [ ] Module 5 — Frontend UI

---

## 🤝 Contributing

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make changes and test with `python main.py`
3. Commit: `git commit -m "feat: describe your change"`
4. Push: `git push origin feature/your-feature-name`
5. Open a Pull Request on GitHub

---

## ⚠️ Notes for Teammates

- **Do NOT commit the `venv/` folder** — it's in `.gitignore`
- **Whisper model** (~150 MB) is downloaded automatically on first run — no manual download needed
- **IndicBERT** (`ai4bharat/indic-bert`) is a gated model. To use it: set `INDICBERT_LOCAL_PATH` in `NLP/intent_recognition.py` to your fine-tuned local folder
- Every line in every file has a `#` comment explaining what it does — this is intentional for learning purposes
