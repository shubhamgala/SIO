# SIO Voice-First Banking Assistant - Automated Master Test Suite               # File title comment
# Runs comprehensive test cases across all modules: STT, NLP, Dialogue, Auth & Banking # File description comment
# Produces formatted test reports for academic evaluations and progress reviews. # File purpose comment

import os  # Import os module for file path operations
import sys  # Import sys module for system paths and encoding setup
import tempfile  # Import tempfile module for temporary test artifacts
import numpy as np  # Import numpy for numerical and array test data

# Reconfigure stdout to UTF-8 for Hindi characters in terminal                    # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if stdout supports reconfigure
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout encoding to UTF-8

# Ensure project root is in Python module search path                             # Description comment
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get project root folder path
if ROOT_DIR not in sys.path:  # Check if root is not in sys.path
    sys.path.insert(0, ROOT_DIR)  # Add root directory to sys.path

# Global test tracking counters                                                   # Description comment
TOTAL_TESTS = 0  # Counter for total test cases executed
PASSED_TESTS = 0  # Counter for passed test cases
FAILED_TESTS = 0  # Counter for failed test cases
TEST_RECORDS = []  # Detailed records list for summary table

def record_test(test_id, module, description, expected, actual, passed):  # Define test reporting helper
    global TOTAL_TESTS, PASSED_TESTS, FAILED_TESTS  # Access global counters
    TOTAL_TESTS += 1  # Increment total count
    if passed:  # If test passed
        PASSED_TESTS += 1  # Increment pass count
        status = "PASS"  # Set status string
    else:  # If test failed
        FAILED_TESTS += 1  # Increment fail count
        status = "FAIL"  # Set status string
    TEST_RECORDS.append({  # Add record dictionary
        "id": test_id, "module": module, "desc": description,  # Record metadata
        "expected": str(expected), "actual": str(actual), "status": status  # Results
    })  # End of append
    symbol = "✓" if passed else "✗"  # Choose terminal display symbol
    print(f"  [{status}] {symbol} {test_id}: {description}")  # Print status line
    if not passed:  # If failed print diff
        print(f"         Expected: {expected}")  # Show expected
        print(f"         Actual  : {actual}")  # Show actual

# ── Module 1 Tests: Speech-to-Text & Audio Signal Processing ──────────────────── #
def test_module_stt():  # Define STT test suite function
    print("\n" + "=" * 75)  # Section header border
    print("  TEST SUITE 1: Module 1 - Speech-to-Text & Audio Processing")  # Section title
    print("=" * 75)  # Section header border
    from STT.whisper_stt import post_process_transcription, apply_audio_filter  # Import STT utilities
    
    # Test TC-STT-01: Leading filler word removal
    res = post_process_transcription("And Raju, send 5000")  # Test input
    record_test("TC-STT-01", "STT", "Remove leading filler 'And'", "Raju, send 5000", res, res == "Raju, send 5000")  # Record
    
    # Test TC-STT-02: Currency formatting cleanup
    res = post_process_transcription("Rs.5,000 to Amit")  # Test input
    record_test("TC-STT-02", "STT", "Normalize 'Rs.5,000' to plain digits", "5000 to Amit", res, res == "5000 to Amit")  # Record
    
    # Test TC-STT-03: Indian comma grouping cleanup
    res = post_process_transcription("Send 50,000 rupees")  # Test input
    record_test("TC-STT-03", "STT", "Remove Indian comma from '50,000'", "Send 50000 rupees", res, res == "Send 50000 rupees")  # Record
    
    # Test TC-STT-04: Trailing currency noise removal
    res = post_process_transcription("Send Raju 5000 rupees to Rs.")  # Test input
    record_test("TC-STT-04", "STT", "Remove trailing noise 'to Rs.'", "Send Raju 5000 rupees", res, res == "Send Raju 5000 rupees")  # Record
    
    # Test TC-STT-05: High-pass audio filtering
    dummy_signal = np.ones(1600, dtype=np.float32)  # DC constant signal
    filtered = apply_audio_filter(dummy_signal)  # Apply filter
    dc_attenuated = float(np.mean(filtered[-400:])) < 0.1  # DC offset must be removed
    record_test("TC-STT-05", "STT", "Highpass filter attenuates DC / low hum", True, dc_attenuated, dc_attenuated)  # Record

# ── Module 2 Tests: NLP Intent & Entity Recognition ──────────────────────────── #
def test_module_nlp():  # Define NLP test suite function
    print("\n" + "=" * 75)  # Section header border
    print("  TEST SUITE 2: Module 2 - Intent & Entity Recognition (Multilingual)")  # Section title
    print("=" * 75)  # Section header border
    from NLP.intent_recognition import process_query  # Import NLP processor
    
    # Test cases mapping: (id, query, expected_intent, expected_amount, expected_recipient)
    nlp_cases = [  # List of intent and entity test cases
        ("TC-NLP-01", "मेरा बैलेंस बताओ", "Balance Enquiry", None, None),  # Hindi balance
        ("TC-NLP-01b", "मेरा बेलेंस कितना है?", "Balance Enquiry", None, None),  # Whisper phonetic balance
        ("TC-NLP-02", "What is my account balance?", "Balance Enquiry", None, None),  # English balance
        ("TC-NLP-03", "Mera balance kitna hai batao", "Balance Enquiry", None, None),  # Hinglish balance
        ("TC-NLP-04", "खाते में कितने पैसे हैं", "Balance Enquiry", None, None),  # Hindi colloquial balance
        ("TC-NLP-05", "रमेश को 500 रुपये ट्रांसफर कर दो", "Fund Transfer", 500, "रमेश"),  # Hindi transfer
        ("TC-NLP-06", "Transfer 1000 rupees to Shubham", "Fund Transfer", 1000, "Shubham"),  # English transfer
        ("TC-NLP-07", "Amit ko 250 rs send karo", "Fund Transfer", 250, "Amit"),  # Hinglish transfer
        ("TC-NLP-08", "Send Raju 5000 rupees", "Fund Transfer", 5000, "Raju"),  # Direct send pattern
        ("TC-NLP-09", "मुझे पैसे भेजने हैं", "Fund Transfer", None, None),  # Hindi transfer intent without slots
        ("TC-NLP-10", "पिछला लेनदेन और स्टेटमेंट दिखाओ", "Transaction History", None, None),  # Hindi history
        ("TC-NLP-10b", "पिछले लेन देन दिखाओ", "Transaction History", None, None),  # Hindi history with spaced len-den
        ("TC-NLP-10c", "स्टेटमेंट दिखाओ", "Transaction History", None, None),  # Hindi statement request
        ("TC-NLP-11", "Show my last 5 transactions", "Transaction History", None, None),  # English history
        ("TC-NLP-12", "Phone me 199 ka recharge kar do", "Mobile Recharge", 199, None),  # Hinglish recharge
        ("TC-NLP-13", "Bijli ka bill bharna hai 1200", "Bill Payment", 1200, None),  # Hinglish bill payment
        ("TC-NLP-14", "नजदीकी एटीएम कहां है?", "ATM Locator", None, None),  # Hindi ATM locator
        ("TC-NLP-15", "Where is the nearest ATM branch?", "ATM Locator", None, None),  # English ATM locator
        ("TC-NLP-16", "Mera ATM card kho gaya hai madad karo", "Help", None, None),  # Card loss / emergency
        ("TC-NLP-17", "कस्टमर केयर से बात करनी है", "Help", None, None),  # Explicit customer support
        ("TC-NLP-18", "नमस्ते", "Unknown", None, None),  # Pure greeting should not trigger false action
    ]  # End of nlp_cases list
    
    for tc_id, query, exp_intent, exp_amount, exp_recipient in nlp_cases:  # Loop through cases
        res = process_query(query)  # Execute NLP processing
        intent_match = res["intent"] == exp_intent  # Check intent
        amount_match = res["entities"].get("amount") == exp_amount  # Check amount
        recip_match = (exp_recipient is None and res["entities"].get("recipient") is None) or \
                      (exp_recipient is not None and res["entities"].get("recipient") == exp_recipient)  # Check recipient
        passed = intent_match and amount_match and recip_match  # Overall pass flag
        desc = f"Classify '{query}' -> {exp_intent}"  # Description
        actual_summary = f"{res['intent']}, amt={res['entities'].get('amount')}, recip={res['entities'].get('recipient')}"  # Actual summary
        exp_summary = f"{exp_intent}, amt={exp_amount}, recip={exp_recipient}"  # Expected summary
        record_test(tc_id, "NLP", desc, exp_summary, actual_summary, passed)  # Record test

# ── Module 3 Tests: Dialogue Management & Confirmation Logic ──────────────────── #
def test_module_dialogue():  # Define Dialogue Manager test suite
    print("\n" + "=" * 75)  # Section header border
    print("  TEST SUITE 3: Module 3 - Dialogue Manager & Slot-Filling Logic")  # Section title
    print("=" * 75)  # Section header border
    from Backend.dialogue_manager import (  # Import DM utilities
        detect_yes_no, extract_amount_from_text, extract_recipient_from_text,
        extract_bill_type_from_text, build_confirmation_message, REQUIRED_SLOTS,
        int_to_hindi, format_numbers_for_hindi_speech  # Import Hindi speech number converter utilities
    )  # End of imports
    
    # Test TC-DM-01 to 05: Confirmation word detection
    record_test("TC-DM-01", "Dialogue", "Detect Hindi 'haan' as confirmation", "yes", detect_yes_no("haan kar do"), detect_yes_no("haan kar do") == "yes")  # Test
    record_test("TC-DM-01b", "Dialogue", "Detect Whisper phonetics 'Ah!' as confirmation", "yes", detect_yes_no("Ah!"), detect_yes_no("Ah!") == "yes")  # Test
    record_test("TC-DM-01c", "Dialogue", "Detect Devanagari 'हाँ' as confirmation", "yes", detect_yes_no("हाँ"), detect_yes_no("हाँ") == "yes")  # Test
    record_test("TC-DM-02", "Dialogue", "Detect English 'yes' as confirmation", "yes", detect_yes_no("yes please confirm"), detect_yes_no("yes please confirm") == "yes")  # Test
    record_test("TC-DM-03", "Dialogue", "Detect Hindi 'nahi' as cancellation", "no", detect_yes_no("nahi mat karo"), detect_yes_no("nahi mat karo") == "no")  # Test
    record_test("TC-DM-03b", "Dialogue", "Detect Devanagari 'नहीं' as cancellation", "no", detect_yes_no("नहीं"), detect_yes_no("नहीं") == "no")  # Test
    record_test("TC-DM-04", "Dialogue", "Detect English 'cancel' as cancellation", "no", detect_yes_no("cancel this payment"), detect_yes_no("cancel this payment") == "no")  # Test
    record_test("TC-DM-05", "Dialogue", "Unclear input classified as unknown", "unknown", detect_yes_no("random text"), detect_yes_no("random text") == "unknown")  # Test
    
    # Test TC-DM-06 to 09: Slot parsing helpers
    amt = extract_amount_from_text("5000 rupaye")  # Digits amount
    record_test("TC-DM-06", "Dialogue", "Extract digits amount from follow-up", 5000, amt, amt == 5000)  # Test
    
    amt_spoken = extract_amount_from_text("ek hazaar rupaye")  # Spoken Hindi amount
    record_test("TC-DM-07", "Dialogue", "Extract spoken Hindi 'ek hazaar' as 1000", 1000, amt_spoken, amt_spoken == 1000)  # Test
    
    recip = extract_recipient_from_text("Raju ko bhejna")  # Recipient follow-up
    record_test("TC-DM-08", "Dialogue", "Extract recipient 'Raju' from follow-up", "Raju", recip, recip == "Raju")  # Test
    
    bill = extract_bill_type_from_text("bijli ka bill bharna hai")  # Bill type follow-up
    record_test("TC-DM-09", "Dialogue", "Extract 'Electricity' bill type", "Electricity", bill, bill == "Electricity")  # Test
    
    # Test TC-DM-10 to 12: Confirmation messages & slots
    msg_transfer = build_confirmation_message("Fund Transfer", {"recipient": "Raju", "amount": 500})  # Build msg
    transfer_ok = ("Raju" in msg_transfer) and ("500" in msg_transfer) and ("Haan ya Nahi" in msg_transfer)  # Check
    record_test("TC-DM-10", "Dialogue", "Build Hindi Fund Transfer confirmation message", True, transfer_ok, transfer_ok)  # Test
    
    msg_balance = build_confirmation_message("Balance Enquiry", {})  # Build msg
    balance_ok = "account balance" in msg_balance and "Haan ya Nahi" in msg_balance  # Check
    record_test("TC-DM-11", "Dialogue", "Build Hindi Balance Enquiry confirmation message", True, balance_ok, balance_ok)  # Test
    
    msg_bill = build_confirmation_message("Bill Payment", {"bill_type": "Electricity", "amount": 1200})  # Build bill confirmation
    bill_msg_ok = "Aapka Electricity bill 1200 rupaye ka hai" in msg_bill and "Haan ya Nahi" in msg_bill  # Verify message text
    record_test("TC-DM-11b", "Dialogue", "Build Hindi Bill Payment confirmation prompt (Aapka bill 1200 ka hai)", True, bill_msg_ok, bill_msg_ok)  # Test
    
    slots_ok = REQUIRED_SLOTS["Fund Transfer"] == ["recipient", "amount"] and REQUIRED_SLOTS["Balance Enquiry"] == []  # Check
    record_test("TC-DM-12", "Dialogue", "Verify required slot definitions for intents", True, slots_ok, slots_ok)  # Test
    
    # Test TC-DM-13 to 15: Hindi number to words spoken converter
    h_500 = int_to_hindi(500)  # Convert 500 to Hindi words
    record_test("TC-DM-13", "Dialogue", "Convert 500 to authentic Hindi 'पाँच सौ'", "पाँच सौ", h_500, h_500 == "पाँच सौ")  # Test
    
    spoken_prompt = format_numbers_for_hindi_speech("Kya aap Raju ko 500 rupaye bhejna chahte hain?")  # Spoken prompt
    prompt_ok = "पाँच सौ" in spoken_prompt and "500" not in spoken_prompt  # Check conversion
    record_test("TC-DM-14", "Dialogue", "TTS preprocessor converts 500 to 'पाँच सौ' in confirmation prompt", True, prompt_ok, prompt_ok)  # Test
    
    spoken_bal = format_numbers_for_hindi_speech("Aapka balance 25,000.00 rupaye hai.")  # Spoken ledger balance
    bal_ok = "पच्चीस हज़ार" in spoken_bal and "25,000" not in spoken_bal  # Check ledger conversion
    record_test("TC-DM-15", "Dialogue", "TTS preprocessor converts 25,000.00 to 'पच्चीस हज़ार' in balance prompt", True, bal_ok, bal_ok)  # Test

# ── Module 4 Tests: Biometric Authentication Configuration & Storage ─────────── #
def test_module_auth():  # Define Auth module test suite
    print("\n" + "=" * 75)  # Section header border
    print("  TEST SUITE 4: Module 4 - Biometric Authentication Storage & Config")  # Section title
    print("=" * 75)  # Section header border
    from auth.config import AuthConfig  # Import AuthConfig
    from auth.storage import save_embedding, load_embedding  # Import embedding storage
    
    # Test TC-AUTH-01: Configuration parameters
    cfg = AuthConfig()  # Create config
    cfg_ok = cfg.face_similarity_threshold == 0.6 and cfg.voice_similarity_threshold == 0.75  # Check thresholds
    record_test("TC-AUTH-01", "Auth", "Validate default biometric similarity thresholds", True, cfg_ok, cfg_ok)  # Test
    
    # Test TC-AUTH-02: Embedding storage save and load roundtrip
    test_user = "test_eval_user"  # Test user id
    dummy_embedding = np.array([0.12, 0.34, 0.56, 0.78], dtype=np.float32)  # Dummy vector
    try:  # Try save and load
        save_embedding(test_user, "face", dummy_embedding)  # Save embedding
        loaded = load_embedding(test_user, "face")  # Load embedding
        roundtrip_ok = np.allclose(dummy_embedding, loaded)  # Check equality
        record_test("TC-AUTH-02", "Auth", "Biometric face embedding save & load roundtrip", True, roundtrip_ok, roundtrip_ok)  # Test
        # Clean up test artifact
        from auth.storage import _embedding_path  # Import path builder
        test_file = _embedding_path(test_user, "face")  # Get path
        if test_file.exists():  # If exists
            test_file.unlink()  # Delete temp test file
    except Exception as e:  # On error
        record_test("TC-AUTH-02", "Auth", "Biometric face embedding save & load roundtrip", True, f"Error: {e}", False)  # Record fail
    
    # Test TC-AUTH-03: Security path traversal rejection
    invalid_user_caught = False  # Flag
    try:  # Try invalid user
        save_embedding("../hacker", "face", dummy_embedding)  # Attempt path traversal
    except ValueError:  # Expected ValueError
        invalid_user_caught = True  # Caught
    record_test("TC-AUTH-03", "Auth", "Reject path traversal attack in user_id", True, invalid_user_caught, invalid_user_caught)  # Test

# ── Module 5 Tests: Core Banking Engine & Database Ledger ─────────────────────── #
def test_module_banking():  # Define Core Banking test suite
    print("\n" + "=" * 75)  # Section header border
    print("  TEST SUITE 5: Module 4 - Core Banking Engine & Database Ledger")  # Section title
    print("=" * 75)  # Section header border
    from Backend.banking_logic import (  # Import banking logic functions
        init_banking_database, check_balance, transfer_funds,
        get_transaction_history, execute_mobile_recharge, pay_utility_bill,
        locate_nearest_atms, handle_help_and_security, execute_banking_action,
        get_db_connection, DEFAULT_ACCOUNT_NUM
    )  # End of imports
    
    init_banking_database()  # Ensure database tables and initial seed data exist
    # Reset test account balance to ₹25,000 and card status to ACTIVE for reproducibility # Test setup
    test_conn = get_db_connection()  # Open test database connection
    test_cursor = test_conn.cursor()  # Create cursor
    test_cursor.execute("UPDATE accounts SET balance = 25000.0, card_status = 'ACTIVE' WHERE account_number = ?", (DEFAULT_ACCOUNT_NUM,))  # Reset test funds
    test_conn.commit()  # Commit reset balance
    test_conn.close()  # Close test connection
    
    # Test TC-BANK-01: Check Balance query returns positive balance
    bal_res = check_balance(DEFAULT_ACCOUNT_NUM)  # Query account balance
    bal_ok = bal_res["status"] == "success" and bal_res["balance"] > 0 and "Shubham" in bal_res.get("holder", "")  # Verify
    record_test("TC-BANK-01", "Banking", "Check balance returns positive balance for mock account", True, bal_ok, bal_ok)  # Test
    
    # Test TC-BANK-02: Fund Transfer with Before and After Balance Verification
    init_bal = check_balance(DEFAULT_ACCOUNT_NUM)["balance"]  # Check starting balance
    tx_res = transfer_funds("Raju", 500.0, DEFAULT_ACCOUNT_NUM)  # Send ₹500 to Raju
    tx_ok = (  # Check transfer math and clean receipt formatting
        tx_res["status"] == "success" and  # Success status
        tx_res["amount"] == 500.0 and  # Amount matched
        tx_res["balance_before"] == init_bal and  # Before balance matched
        abs(tx_res["balance_after"] - (init_bal - 500.0)) < 0.01 and  # Math verified
        "Raju" in tx_res["recipient"] and  # Recipient verified
        tx_res["display"].count("FUNDS TRANSFER RECEIPT") == 1 and  # Receipt header appears exactly once
        len(tx_res["display"]) < 1000  # Receipt card is compact and not bloated by operator multiplication
    )  # End of transfer condition
    record_test("TC-BANK-02", "Banking", "Transfer ₹500 to Raju validates Balance BEFORE - 500 == AFTER and clean receipt", True, tx_ok, tx_ok)  # Test
    
    # Test TC-BANK-03: Insufficient Funds handling
    insuf_res = transfer_funds("Amit", 99999999.0, DEFAULT_ACCOUNT_NUM)  # Try transferring huge amount
    insuf_ok = insuf_res["status"] == "insufficient_funds" and "paryapt balance" in insuf_res["speech"].lower()  # Verify
    record_test("TC-BANK-03", "Banking", "Overdraft rejection prevents transfer exceeding balance", True, insuf_ok, insuf_ok)  # Test
    
    # Test TC-BANK-04: Transaction History Passbook retrieval
    hist_res = get_transaction_history(5, DEFAULT_ACCOUNT_NUM)  # Request last 5 transactions
    hist_ok = hist_res["status"] == "success" and hist_res.get("count", 0) >= 5 and "PASSBOOK" in hist_res["display"]  # Verify
    record_test("TC-BANK-04", "Banking", "Transaction history retrieves 5 transactions in passbook table", True, hist_ok, hist_ok)  # Test
    
    # Test TC-BANK-05: Mobile Recharge Before & After balance
    pre_rec_bal = check_balance(DEFAULT_ACCOUNT_NUM)["balance"]  # Starting balance
    rec_res = execute_mobile_recharge(199.0, "9876543210", DEFAULT_ACCOUNT_NUM)  # Recharge ₹199
    rec_ok = rec_res["status"] == "success" and abs(rec_res["balance_after"] - (pre_rec_bal - 199.0)) < 0.01  # Math check
    record_test("TC-BANK-05", "Banking", "Mobile recharge ₹199 deducts accurately with before/after audit", True, rec_ok, rec_ok)  # Test
    
    # Test TC-BANK-06: Utility Bill Payment Status update
    bill_res = pay_utility_bill("Electricity", 1200.0, DEFAULT_ACCOUNT_NUM)  # Pay electricity bill
    bill_ok = bill_res["status"] == "success" and "PAID" in bill_res["display"] and "Electricity" in bill_res["speech"]  # Check
    record_test("TC-BANK-06", "Banking", "Electricity bill payment executes and marks bill as PAID", True, bill_ok, bill_ok)  # Test
    
    # Test TC-BANK-07: ATM Locator returns nearest branches with wheelchair & cash
    atm_res = locate_nearest_atms()  # Locate ATMs
    atm_ok = atm_res["status"] == "success" and atm_res.get("count", 0) >= 3 and "ATMS & BRANCHES" in atm_res["display"]  # Check
    record_test("TC-BANK-07", "Banking", "ATM Locator returns nearby branches with distance and accessibility", True, atm_ok, atm_ok)  # Test
    
    # Test TC-BANK-08: Emergency Debit Card Blocking updates DB status to BLOCKED
    block_res = handle_help_and_security("lost card please block", DEFAULT_ACCOUNT_NUM)  # Request card block
    conn = get_db_connection()  # Connect to database
    cursor = conn.cursor()  # Get cursor
    cursor.execute("SELECT card_status FROM accounts WHERE account_number = ?", (DEFAULT_ACCOUNT_NUM,))  # Query card
    card_st = cursor.fetchone()["card_status"]  # Get status
    conn.close()  # Close connection
    card_ok = block_res["status"] == "card_blocked" and card_st == "BLOCKED"  # Verify status
    record_test("TC-BANK-08", "Banking", "Emergency card loss inquiry marks Debit Card as BLOCKED in DB", True, card_ok, card_ok)  # Test
    
    # Test TC-BANK-09: Unified Dispatcher for Balance Enquiry
    action_res = execute_banking_action({"intent": "Balance Enquiry", "entities": {}}, DEFAULT_ACCOUNT_NUM)  # Dispatch
    action_ok = action_res["status"] == "success" and "balance" in action_res  # Verify
    record_test("TC-BANK-09", "Banking", "Unified dispatcher routes 'Balance Enquiry' correctly", True, action_ok, action_ok)  # Test

# ── Main Runner Function ──────────────────────────────────────────────────────── #
def run_all():  # Define master runner
    print("=" * 75)  # Header top
    print("   SIO — SMART INDIAN BANKING ASSISTANT: MASTER TEST SUITE")  # Header title
    print("   Academic Evaluation & Verification Test Runner")  # Header subtitle
    print("=" * 75)  # Header bottom
    
    test_module_stt()  # Execute STT tests
    test_module_nlp()  # Execute NLP tests
    test_module_dialogue()  # Execute Dialogue tests
    test_module_auth()  # Execute Auth tests
    test_module_banking()  # Execute Core Banking tests
    
    # Summary calculation
    pass_percentage = (PASSED_TESTS / TOTAL_TESTS * 100) if TOTAL_TESTS > 0 else 0.0  # Calculate %
    
    print("\n" + "=" * 75)  # Summary top
    print("                      COMPREHENSIVE TEST SUMMARY")  # Summary title
    print("=" * 75)  # Summary border
    print(f"  Total Test Cases Executed : {TOTAL_TESTS}")  # Total
    print(f"  Passed                    : {PASSED_TESTS}  ✓")  # Passed
    print(f"  Failed                    : {FAILED_TESTS}  ✗")  # Failed
    print(f"  Success Rate              : {pass_percentage:.1f}%")  # Percentage
    print("=" * 75)  # Summary border
    
    if FAILED_TESTS == 0:  # If all passed
        print("  🎉 ALL TEST CASES PASSED SUCCESSFULLY — READY FOR REVIEW!")  # Success message
    else:  # If any failed
        print(f"  ⚠️  {FAILED_TESTS} TEST CASE(S) FAILED. CHECK LOGS ABOVE.")  # Warning message
    print("=" * 75 + "\n")  # Footer border

if __name__ == "__main__":  # If script executed directly
    run_all()  # Run all test suites
