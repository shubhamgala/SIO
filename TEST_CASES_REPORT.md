# SIO: Smart Indian Banking Voice Assistant
## Formal Software Test Case Specification & Verification Report

**Project Title:** SIO — Multilingual Voice-First Banking Assistant for Elderly Users  
**Version:** 1.0.0 (Development Phase)  
**Execution Date:** 2026-10-15  
**Test Runner Script:** `run_all_tests.py`  
**Overall Result:** **53 / 53 Tests Passed (100.0% Pass Rate)**  

---

### Executive Test Summary

| Module Name | Component / Scope | Tests Executed | Passed | Failed | Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **Module 1 (STT)** | Audio Filtering, Hallucination Cleanup & Currency Normalization | 5 | 5 | 0 | **PASS** |
| **Module 2 (NLP)** | Multilingual Intent Classification & Financial Entity Extraction | 18 | 18 | 0 | **PASS** |
| **Module 3 (Dialogue)** | Slot-Filling, Phonetic Yes/No, Devanagari Affirmatives & Hindi Number TTS | 18 | 18 | 0 | **PASS** |
| **Module 4 (Auth)** | Biometric Thresholds, Vector Storage & Path Traversal Security | 3 | 3 | 0 | **PASS** |
| **Module 4 (Core Banking)** | SQLite Ledger, Balance Before/After, Passbook, Bills, Recharge, ATM & Security | 9 | 9 | 0 | **PASS** |
| **Total** | **End-to-End System Integration** | **53** | **53** | **0** | **100% PASS** |

---

### Detailed Test Case Execution Matrix

#### Module 1: Speech-to-Text & Audio Signal Processing (`STT/whisper_stt.py`)

| Test ID | Test Objective | Input | Expected Output | Actual Output | Status |
|:---|:---|:---|:---|:---|:---:|
| `TC-STT-01` | Remove leading filler word hallucinated by STT | `"And Raju, send 5000"` | `"Raju, send 5000"` | `"Raju, send 5000"` | **PASS** |
| `TC-STT-02` | Normalize currency prefix with commas | `"Rs.5,000 to Amit"` | `"5000 to Amit"` | `"5000 to Amit"` | **PASS** |
| `TC-STT-03` | Normalize standalone Indian comma grouped numbers | `"Send 50,000 rupees"` | `"Send 50000 rupees"` | `"Send 50000 rupees"` | **PASS** |
| `TC-STT-04` | Strip trailing currency noise artifacts | `"Send Raju 5000 rupees to Rs."` | `"Send Raju 5000 rupees"` | `"Send Raju 5000 rupees"` | **PASS** |
| `TC-STT-05` | Highpass Butterworth filter (80 Hz) attenuation | Constant DC + 50Hz hum signal | Attenuation > 90% (mean < 0.1) | Attenuated (mean < 0.05) | **PASS** |

---

#### Module 2: NLP Intent & Entity Recognition (`NLP/intent_recognition.py`)

| Test ID | Language & Category | Test Query | Expected Intent | Extracted Entities | Status |
|:---|:---|:---|:---|:---|:---:|
| `TC-NLP-01` | Hindi Balance Enquiry | `"मेरा बैलेंस बताओ"` | `Balance Enquiry` | `None` | **PASS** |
| `TC-NLP-02` | English Balance Enquiry | `"What is my account balance?"` | `Balance Enquiry` | `None` | **PASS** |
| `TC-NLP-03` | Hinglish Balance Enquiry | `"Mera balance kitna hai batao"` | `Balance Enquiry` | `None` | **PASS** |
| `TC-NLP-04` | Hindi Colloquial Balance | `"खाते में कितने पैसे हैं"` | `Balance Enquiry` | `None` | **PASS** |
| `TC-NLP-05` | Hindi Fund Transfer | `"रमेश को 500 रुपये ट्रांसफर कर दो"` | `Fund Transfer` | `amt: 500, recip: रमेश` | **PASS** |
| `TC-NLP-06` | English Fund Transfer | `"Transfer 1000 rupees to Shubham"` | `Fund Transfer` | `amt: 1000, recip: Shubham` | **PASS** |
| `TC-NLP-07` | Hinglish Fund Transfer | `"Amit ko 250 rs send karo"` | `Fund Transfer` | `amt: 250, recip: Amit` | **PASS** |
| `TC-NLP-08` | Direct Verb-Name Pattern | `"Send Raju 5000 rupees"` | `Fund Transfer` | `amt: 5000, recip: Raju` | **PASS** |
| `TC-NLP-09` | Hindi Transfer (Slotless) | `"मुझे पैसे भेजने हैं"` | `Fund Transfer` | `None (triggers slot fill)` | **PASS** |
| `TC-NLP-10` | Hindi Statement Request | `"पिछला लेनदेन और स्टेटमेंट दिखाओ"` | `Transaction History` | `None` | **PASS** |
| `TC-NLP-11` | English Mini-Statement | `"Show my last 5 transactions"` | `Transaction History` | `None (excludes count)` | **PASS** |
| `TC-NLP-12` | Hinglish Mobile Recharge | `"Phone me 199 ka recharge kar do"` | `Mobile Recharge` | `amt: 199` | **PASS** |
| `TC-NLP-13` | Hinglish Utility Payment | `"Bijli ka bill bharna hai 1200"` | `Bill Payment` | `amt: 1200` | **PASS** |
| `TC-NLP-14` | Hindi ATM Locator | `"नजदीकी एटीएम कहां है?"` | `ATM Locator` | `None` | **PASS** |
| `TC-NLP-15` | English Branch Locator | `"Where is the nearest ATM branch?"` | `ATM Locator` | `None` | **PASS** |
| `TC-NLP-16` | Card Loss / Emergency | `"Mera ATM card kho gaya hai madad karo"` | `Help` | `None` | **PASS** |
| `TC-NLP-17` | Customer Care Request | `"कस्टमर केयर से बात करनी है"` | `Help` | `None` | **PASS** |
| `TC-NLP-18` | Standalone Greeting Safety | `"नमस्ते"` | `Unknown` | `None (graceful repeat)` | **PASS** |

---

#### Module 3: Dialogue Manager & State Machine (`Backend/dialogue_manager.py`)

| Test ID | Test Objective | Spoken Input | Expected Interpretation | Status |
|:---|:---|:---|:---|:---:|
| `TC-DM-01` | Hindi Confirmation Detection | `"haan kar do"` | `yes` (Proceed action) | **PASS** |
| `TC-DM-01b` | Whisper Phonetic Confirmation | `"Ah!"` | `yes` (Soft affirmative capture) | **PASS** |
| `TC-DM-01c` | Devanagari Hindi Affirmative | `"हाँ"` | `yes` (Script confirmation) | **PASS** |
| `TC-DM-02` | English Confirmation Detection | `"yes please confirm"` | `yes` (Proceed action) | **PASS** |
| `TC-DM-03` | Hindi Cancellation Detection | `"nahi mat karo"` | `no` (Abort action) | **PASS** |
| `TC-DM-03b` | Devanagari Hindi Cancellation | `"नहीं"` | `no` (Script cancellation) | **PASS** |
| `TC-DM-04` | English Cancellation Detection | `"cancel this payment"` | `no` (Abort action) | **PASS** |
| `TC-DM-05` | Unclear / Gibberish Detection | `"random gibberish"` | `unknown` (Prompt retry) | **PASS** |
| `TC-DM-06` | Digits Amount Slot Extraction | `"5000 rupaye"` | `5000` | **PASS** |
| `TC-DM-07` | Hindi Spoken Number Slot | `"ek hazaar rupaye"` | `1000` | **PASS** |
| `TC-DM-08` | Recipient Slot Extraction | `"Raju ko bhejna"` | `"Raju"` | **PASS** |
| `TC-DM-09` | Bill Type Slot Extraction | `"bijli ka bill bharna hai"` | `"Electricity"` | **PASS** |
| `TC-DM-10` | Fund Transfer Readback Prompt | Recipient: Raju, Amount: 500 | `"Kya aap Raju ko 500 rupaye bhejna chahte hain? Haan ya Nahi?"` | **PASS** |
| `TC-DM-11` | Balance Readback Prompt | Balance intent | `"Kya aap apna account balance check karna chahte hain? Haan ya Nahi?"` | **PASS** |
| `TC-DM-12` | Slot Requirement Verification | Fund Transfer vs Balance | `FT: ['recipient', 'amount']`, `BE: []` | **PASS** |
| `TC-DM-13` | Number to Hindi Words Conversion | `500` | `"पाँच सौ"` | **PASS** |
| `TC-DM-14` | Spoken Readback TTS Normalization | `"... 500 rupaye ..."` | `"... पाँच सौ rupaye ..."` | **PASS** |
| `TC-DM-15` | Spoken Ledger Balance Normalization | `"... 25,000.00 rupaye ..."` | `"... पच्चीस हज़ार rupaye ..."` | **PASS** |

---

#### Module 4 (Part A): Biometric Authentication Storage & Config (`auth/`)

| Test ID | Test Objective | Test Input / Action | Expected Result | Status |
|:---|:---|:---|:---|:---:|
| `TC-AUTH-01` | Verify Biometric Thresholds | `AuthConfig()` | `face: 0.6, voice: 0.75, grace: 5s` | **PASS** |
| `TC-AUTH-02` | Biometric Vector Serialization | Save & reload 4D numpy array | `np.allclose(original, loaded) == True` | **PASS** |
| `TC-AUTH-03` | Path Traversal Security Defense | `save_embedding("../hacker", "face")` | `ValueError` (Rejection of directory traversal) | **PASS** |

---

#### Module 4 (Part B): Core Banking Engine & Database Ledger (`Backend/banking_logic.py`)

| Test ID | Test Objective | Test Input / Action | Expected Verification | Status |
|:---|:---|:---|:---|:---:|
| `TC-BANK-01` | Account Balance Retrieval | Query mock account `SB-8839201452` | Positive balance returned for Shubham Gala | **PASS** |
| `TC-BANK-02` | Fund Transfer with Before/After Audit | Send ₹500 to Raju | Exact debit: `Balance BEFORE - 500.00 == Balance AFTER` | **PASS** |
| `TC-BANK-03` | Overdraft Rejection Protection | Attempt transfer of ₹99,999,999 | Rejection with `insufficient_funds` & polite Hindi audio | **PASS** |
| `TC-BANK-04` | Passbook Transaction History | Query last 5 transactions | Formatted ASCII Passbook table with $\ge 5$ entries | **PASS** |
| `TC-BANK-05` | Mobile Prepaid Recharge | Recharge ₹199 to `9876543210` | Exact balance deduction with Before/After ledger audit | **PASS** |
| `TC-BANK-06` | Utility Bill Payment Execution | Pay ₹1,200 Electricity Bill | Bill marked as `PAID` in DB and receipt generated | **PASS** |
| `TC-BANK-07` | Wheelchair-Accessible ATM Locator | Scan nearby ATM branches | Sorted distances, wheelchair ramp notes & cash availability | **PASS** |
| `TC-BANK-08` | Emergency Card Hotlisting / Blocking | Query `"lost card please block"` | Account card status updated to `BLOCKED` in SQLite DB | **PASS** |
| `TC-BANK-09` | Unified Intent Dispatcher | Route `"Balance Enquiry"` | Seamless routing from Dialogue output to Core Banking | **PASS** |

---

### How to Run the Automated Test Suite in Front of Faculty

```powershell
# Open terminal in SIO project folder and run:
.\venv\Scripts\python.exe run_all_tests.py
```
