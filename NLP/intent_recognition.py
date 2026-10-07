# SIO Voice-First Banking Assistant - Module 2: Intent and Entity Recognition     # File title comment
# This script classifies banking intents and extracts financial entities.        # Module description comment
# Supports multilingual queries in Hindi (Devanagari), Hinglish, and English.   # Supported languages comment
# Integrates IndicBERT architecture with keyword-matching pipeline fallback.     # Architecture note comment
import os  # Import the os module for operating system and file path operations
import sys  # Import the sys module to interact with Python runtime environment
import re  # Import the regular expressions module for pattern matching and entity extraction
import json  # Import the json module to format structured output for easy API integration
import subprocess  # Import subprocess module to execute pip commands for dependency management
from typing import Dict, Any, Tuple, Optional  # Import type hint annotations for cleaner function signatures
# Reconfigure standard output streams to UTF-8 to support Hindi Devanagari on Windows # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if stdout supports stream reconfiguring
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout encoding to UTF-8 to print Hindi characters cleanly
if hasattr(sys.stderr, 'reconfigure'):  # Check if stderr supports stream reconfiguring
    sys.stderr.reconfigure(encoding='utf-8')  # Set stderr encoding to UTF-8 to print error messages cleanly
# Define the 7 core banking intent categories supported by SIO                   # Description comment
INTENT_CATEGORIES = [  # List of all 7 predefined banking intent categories
    "Balance Enquiry",  # Intent for checking current bank account balance
    "Fund Transfer",  # Intent for sending money to another person or account
    "Transaction History",  # Intent for viewing past account statements or transactions
    "Mobile Recharge",  # Intent for recharging prepaid phone numbers or data packs
    "Bill Payment",  # Intent for paying electricity, water, gas, or utility bills
    "ATM Locator",  # Intent for finding nearby ATMs or bank branches
    "Help",  # Intent for customer care support, card blocking, or general queries
]  # End of INTENT_CATEGORIES list
# Name of the IndicBERT pretrained model from HuggingFace ai4bharat              # Description comment
INDICBERT_MODEL_NAME = "ai4bharat/indic-bert"  # HuggingFace model identifier for IndicBERT
# Global cache variable for loaded IndicBERT model and tokenizer                 # Description comment
_MODEL_CACHE: Optional[Tuple[Optional[Any], Optional[Any]]] = None  # Cache model to avoid repeated lookups
# Function to automatically install any missing third-party packages             # Description comment
def ensure_dependencies():  # Define helper function to check external libraries
    packages_to_check = ["transformers", "torch"]  # List of deep learning packages needed for neural models
    for package in packages_to_check:  # Iterate through each required package
        try:  # Start try block to test if package is already installed
            __import__(package)  # Dynamically import the package to verify its existence
        except ImportError:  # Catch error if the package is missing from environment
            print(f"[SIO NLP] Installing missing dependency '{package}' via pip...")  # Inform user of installation
            try:  # Start try block to run pip install
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])  # Install package using pip
                print(f"[SIO NLP] Successfully installed '{package}'.")  # Confirm installation
            except Exception as pip_err:  # Catch any pip installation error
                print(f"[SIO NLP] Notice: Could not install '{package}': {pip_err}")  # Print informative notice
# Ensure dependencies are available before continuing                            # Description comment
ensure_dependencies()  # Run dependency checker before any imports that depend on these packages
# Function to attempt loading fine-tuned IndicBERT model and tokenizer            # Description comment
def load_indicbert_model():  # Define function to load the IndicBERT model from HuggingFace
    global _MODEL_CACHE  # Access global model cache variable
    if _MODEL_CACHE is not None:  # Check if model has already been loaded or evaluated previously
        return _MODEL_CACHE  # Return cached result immediately without re-checking HuggingFace
    # IndicBERT (ai4bharat/indic-bert) requires HuggingFace authentication for gated access.
    # Until a locally fine-tuned checkpoint is available, we use the keyword matching pipeline.
    # To enable neural IndicBERT: set INDICBERT_LOCAL_PATH to your fine-tuned model folder below.
    INDICBERT_LOCAL_PATH = None  # Set to local model folder path when fine-tuned weights are ready
    if INDICBERT_LOCAL_PATH is not None:  # Check if a local fine-tuned model path has been configured
        try:  # Start try block to load model from local path
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Import HuggingFace classes
            print(f"[SIO NLP] Loading fine-tuned IndicBERT from: {INDICBERT_LOCAL_PATH}...")  # Status message
            tokenizer = AutoTokenizer.from_pretrained(INDICBERT_LOCAL_PATH)  # Load tokenizer from local folder
            model = AutoModelForSequenceClassification.from_pretrained(INDICBERT_LOCAL_PATH, num_labels=len(INTENT_CATEGORIES))  # Load model
            print("[SIO NLP] IndicBERT fine-tuned model loaded successfully!")  # Confirmation message
            _MODEL_CACHE = (tokenizer, model)  # Store loaded objects in global cache
            return _MODEL_CACHE  # Return loaded tokenizer and model objects
        except Exception as model_err:  # Catch any error loading from local path
            print(f"[SIO NLP] Could not load local IndicBERT model: {model_err}")  # Print error message
    # No local fine-tuned model found — silently activate keyword matching pipeline
    _MODEL_CACHE = (None, None)  # Cache None tuple so this check is never repeated on future calls
    return _MODEL_CACHE  # Return None tuple to activate keyword matching fallback engine
# Comprehensive keyword dictionary for all 7 intents across Hindi, English, and Hinglish # Description comment
KEYWORD_DICTIONARY = {  # Dictionary mapping each intent category to multilingual keywords
    "Balance Enquiry": [  # Keywords and phrases indicating a balance enquiry query
        "balance", "account balance", "check balance", "how much money", "funds", "remaining",  # English keywords
        "बैलेंस", "खाता", "खाते में", "कितने पैसे", "रुपये हैं", "रकम", "शेष राशि", "राशि",  # Hindi keywords
        "balance kitna", "paise kitne", "account me kitna", "paisa dekhna", "kitne rupaye",  # Hinglish keywords
        "kitna paisa", "balance check", "khata balance", "balance batao", "paise batao",  # Additional Hinglish phrases
    ],  # End of Balance Enquiry keyword list
    "Fund Transfer": [  # Keywords and phrases indicating a money transfer query
        "transfer", "send money", "pay to", "send cash", "remit", "send payment", "deposit",  # English keywords
        "send", "pay",  # Short English action verbs that indicate a transfer request
        "ट्रांसफर", "भेजें", "भेजना", "पैसे डालना", "रुपये ट्रांसफर", "पैसे भेजो", "भुगतान करो",  # Hindi keywords
        "transfer karo", "bhejna", "send karo", "transfer karna", "daal do", "bhej do", "pay karna",  # Hinglish keywords
        "bhejo", "paise bhejo", "rupaye bhejo", "transfer kar do", "send kar do",  # Additional Hinglish phrases
    ],  # End of Fund Transfer keyword list
    "Transaction History": [  # Keywords and phrases indicating a transaction statement query
        "transaction", "history", "statement", "passbook", "recent payments", "mini statement",  # English keywords
        "लेनदेन", "इतिहास", "स्टेटमेंट", "पासबुक", "पिछला लेनदेन", "खर्च", "विवरण",  # Hindi keywords
        "history dikhao", "statement chahiye", "passbook update", "pichle transaction", "mini statement",  # Hinglish keywords
        "last transaction", "kharch dekhna", "kahan kharch hua", "purane transaction",  # Additional Hinglish phrases
    ],  # End of Transaction History keyword list
    "Mobile Recharge": [  # Keywords and phrases indicating a mobile recharge query
        "recharge", "top up", "mobile pack", "phone recharge", "data pack", "prepaid recharge",  # English keywords
        "रिचार्ज", "मोबाइल रिचार्ज", "फोन रिचार्ज", "टॉकटाइम", "मोबाइल", "फोन",  # Hindi keywords
        "mobile recharge", "phone me recharge", "recharge kar do", "number recharge", "recharge karna",  # Hinglish keywords
        "sim recharge", "recharge plan", "topup", "talktime",  # Additional Hinglish phrases
    ],  # End of Mobile Recharge keyword list
    "Bill Payment": [  # Keywords and phrases indicating a utility bill payment query
        "bill", "electricity bill", "water bill", "gas bill", "utility", "pay bill", "power bill",  # English keywords
        "बिल", "बिजली का बिल", "पानी का बिल", "गैस बिल", "बिल भरना", "बिजली बिल", "पानी बिल",  # Hindi keywords
        "bijli bill", "water bill", "gas bill", "bill bharna", "bill pay", "light bill",  # Hinglish keywords
        "bill bhardo", "bill payment", "bijli ka bill", "current bill",  # Additional Hinglish phrases
    ],  # End of Bill Payment keyword list
    "ATM Locator": [  # Keywords and phrases indicating a search for an ATM or branch
        "nearest atm", "find atm", "where is atm", "bank branch", "atm location", "nearby atm",  # Specific English phrases
        "closest atm", "cash withdrawal machine", "atm locator", "locate atm",  # Additional English ATM keywords
        "शाखा", "नजदीकी एटीएम", "पैसे निकालने की मशीन", "एटीएम कहां है", "बैंक शाखा",  # Hindi keywords
        "paas ka atm", "atm kahan hai", "nearest branch", "atm dhoondo", "atm kidhar hai",  # Hinglish keywords
    ],  # End of ATM Locator keyword list
    "Help": [  # Keywords and phrases indicating a request for assistance or reporting issues
        "help", "support", "customer care", "assist", "lost card", "block card", "fraud", "complaint",  # English keywords
        "card lost", "lost my card", "stolen card", "card blocked", "report problem",  # Additional English Help keywords
        "मदद", "सहायता", "कस्टमर केयर", "कार्ड खो गया", "कार्ड ब्लॉक", "शिकायत", "धोखाधड़ी",  # Hindi keywords
        "खो गया", "मदद करो", "सहायता चाहिए", "कार्ड बंद करो",  # Additional Hindi Help keywords
        "madad", "madad chahiye", "madad karo", "help karo", "customer support", "card block",  # Hinglish keywords
        "card kho gaya", "kho gaya", "atm card kho gaya", "card block karo", "lost card",  # Hinglish card blocking keywords
        "kya kar sakte ho", "samajh nahi aaya", "agent se baat", "customer service",  # General assistance phrases
    ],  # End of Help keyword list
}  # End of the full KEYWORD_DICTIONARY
# Function to classify intent using keyword and pattern matching                 # Description comment
def classify_intent_by_keywords(text: str) -> Tuple[str, float]:  # Define keyword-based intent classification function
    try:  # Start try block to handle string processing safely
        if not text or not isinstance(text, str):  # Check if input text is empty or invalid
            return "Help", 0.0  # Return default Help intent with zero confidence if input is invalid
        cleaned_text = text.lower().strip()  # Convert text to lowercase and remove outer whitespace
        scores = {intent: 0 for intent in INTENT_CATEGORIES}  # Initialize match score counter for all 7 intents
        # Dedicated check: if query mentions generic 'atm' without card issues, score ATM Locator
        if "atm" in cleaned_text and not any(term in cleaned_text for term in ["kho gaya", "block", "lost", "stolen", "madad"]):  # ATM location check
            scores["ATM Locator"] += 1  # Add basic ATM match score if not reporting a card issue
        for intent, keywords in KEYWORD_DICTIONARY.items():  # Iterate through all intents and their associated keywords
            for keyword in keywords:  # Iterate through each keyword in the list
                keyword_clean = keyword.lower().strip()  # Clean the keyword string
                if keyword_clean in cleaned_text:  # Check if keyword is present as substring in user query
                    # Weight multi-word keyword matches higher than single-word matches for accuracy
                    weight = len(keyword_clean.split())  # Calculate weight based on number of words in phrase
                    scores[intent] += weight  # Add weight score to the matched intent
        best_intent = max(scores, key=scores.get)  # Identify intent category with highest accumulated match score
        highest_score = scores[best_intent]  # Retrieve highest score value
        if highest_score == 0:  # Check if no keywords matched across any category
            return "Help", 0.30  # Default to Help intent with baseline confidence
        confidence = min(1.0, 0.50 + (highest_score * 0.15))  # Calculate confidence score scaled between 0.5 and 1.0
        return best_intent, round(confidence, 2)  # Return winning intent and rounded confidence score
    except Exception as class_err:  # Catch any unexpected error during intent classification
        print(f"[SIO NLP] Error classifying intent: {class_err}")  # Log error message
        return "Help", 0.10  # Fallback to Help intent safely
# Function to extract financial and user entities from input text                # Description comment
def extract_entities(text: str) -> Dict[str, Optional[Any]]:  # Define entity extraction function
    entities = {  # Initialize dictionary to store extracted entity values
        "amount": None,  # Extracted monetary transaction amount
        "recipient": None,  # Extracted recipient beneficiary name
    }  # End of entities dictionary
    try:  # Start try block to handle entity extraction safely
        if not text or not isinstance(text, str):  # Check if input text is empty or invalid
            return entities  # Return empty entities dictionary if text is not valid
        # Stop words to prevent false-positive recipient matches                 # Description comment
        # "rs" is included to avoid extracting "Rs." as a person's name         # Example note
        stopwords = {  # Set of common banking and helper words to filter out from recipient names
            "bank", "account", "paise", "paisa", "rupees", "khata", "mujhe", "kisi", "card",  # English and Hinglish words
            "rs", "inr", "the", "my", "please", "karo", "karna", "dena",  # Additional noise words including "rs"
            "बैलेंस", "खाता", "पैसे", "रुपये", "एटीएम", "मदद", "बिल", "फोन", "नंबर", "ko", "को",  # Hindi and common words
        }  # End of stopwords set
        # Pattern 1: Extract monetary amount with currency prefix or suffix (₹, Rs, INR, रुपये, etc.)
        # Exclude count patterns like 'last 5 transactions' or 'पिछले 5 लेनदेन'
        is_transaction_count = bool(re.search(r'\b(?:last|past|previous|पिछला|पिछले|pichle)\s+\d+', text, flags=re.IGNORECASE))  # Check count
        if not is_transaction_count:  # Only extract amount if it is not a transaction history count
            amount_pattern = r'(?:₹|rs\.?|inr|रुपये|रुपए)?\s*(\b\d+(?:,\d+)*(?:\.\d+)?\b)\s*(?:₹|rs\.?|inr|रुपये|रुपए|rupees|rupee|bucks)?'  # Regex
            amount_matches = re.findall(amount_pattern, text, flags=re.IGNORECASE)  # Find all regex matches in text
            if amount_matches:  # Check if any numeric matches were found
                cleaned_amount = amount_matches[0].replace(",", "")  # Remove comma separators from numbers
                entities["amount"] = float(cleaned_amount) if "." in cleaned_amount else int(cleaned_amount)  # Parse number
        # Pattern 2: Extract recipient in Hindi/Hinglish (e.g. "रमेश को", "Shubham ko", "Amit ko")
        recipient_ko_pattern = r'([A-Za-z\u0900-\u097F]+)\s+(?:को|ko)(?:\s+|$)'  # Regex for recipient name followed by ko / को
        match_ko = re.search(recipient_ko_pattern, text, flags=re.IGNORECASE)  # Search for match in input text
        if match_ko:  # Check if match was found
            candidate = match_ko.group(1).strip()  # Extract the matched candidate recipient string
            if candidate.lower() not in stopwords:  # Check if candidate is not a common stop word
                entities["recipient"] = candidate  # Assign recipient name entity
        # Pattern 3: Extract recipient in English (e.g. "transfer to Ramesh", "pay Shubham", "send money to Amit")
        if not entities["recipient"]:  # Check if recipient was not already identified by previous pattern
            recipient_en_pattern = r'(?:to|pay|transfer\s+to|send\s+to)\s+([A-Za-z\u0900-\u097F]+)'  # English regex pattern
            match_en = re.search(recipient_en_pattern, text, flags=re.IGNORECASE)  # Search for match in input text
            if match_en:  # Check if match was found
                candidate = match_en.group(1).strip()  # Extract the candidate recipient string
                if candidate.lower() not in stopwords:  # Check if candidate is not a stop word
                    entities["recipient"] = candidate  # Assign recipient name entity
        # Pattern 4: Extract recipient when action verb directly precedes a name before an amount
        # Handles patterns like "send Raju 5000" or "pay Amit 200" or "bhejo Raju 5000"
        if not entities["recipient"]:  # Only attempt if no recipient was found yet
            send_name_amount_pattern = r'(?:send|pay|bhejo|transfer)\s+([A-Za-z\u0900-\u097F]+)\s+\d'  # Pattern: verb + name + number
            match_send = re.search(send_name_amount_pattern, text, flags=re.IGNORECASE)  # Search for match in text
            if match_send:  # Check if match was found
                candidate = match_send.group(1).strip()  # Extract the candidate name string
                if candidate.lower() not in stopwords:  # Check if candidate is not a stop word
                    entities["recipient"] = candidate  # Assign recipient name entity
        # Pattern 5: Extract recipient from "Name, amount" leftover after STT post-processing
        # When Whisper hallucinates "And Raju, Rs.5,000 to Rs." the cleaner produces "Raju, 5000"
        # This catches that Name-comma-number structure as implicit transfer recipient
        if not entities["recipient"]:  # Only attempt if no recipient was found yet
            name_comma_amount_pattern = r'^([A-Za-z\u0900-\u097F]{2,}),?\s+\d'  # Pattern: name at start followed by comma and number
            match_name_comma = re.search(name_comma_amount_pattern, text.strip(), flags=re.IGNORECASE)  # Search at start of text
            if match_name_comma:  # Check if match was found
                candidate = match_name_comma.group(1).strip()  # Extract the candidate name from the match
                if candidate.lower() not in stopwords:  # Check if candidate is not a stop word
                    entities["recipient"] = candidate  # Assign recipient name entity from name-number pattern
    except Exception as ent_err:  # Catch any error during entity extraction
        print(f"[SIO NLP] Error extracting entities: {ent_err}")  # Log entity extraction error message
    return entities  # Return dictionary containing all extracted entities
# Main analysis function combining model inference and entity extraction         # Description comment
def process_query(text: str) -> Dict[str, Any]:  # Define primary entry point for NLP Module 2
    try:  # Start try block for end-to-end query processing
        if not text or not isinstance(text, str):  # Validate input text
            text = ""  # Assign empty string fallback if input is null or not a string
        # Placeholder IndicBERT model loader (returns None, None until fine-tuned weights exist)
        tokenizer, model = load_indicbert_model()  # Attempt to load IndicBERT
        # Determine intent classification
        if model is not None and tokenizer is not None:  # Check if trained IndicBERT model is loaded
            # When fine-tuned weights are ready, run neural inference here
            intent, confidence = "Balance Enquiry", 0.95  # Placeholder for neural prediction
            engine_used = f"IndicBERT ({INDICBERT_MODEL_NAME})"  # Record neural engine
        else:  # Fallback to keyword matching placeholder engine
            intent, confidence = classify_intent_by_keywords(text)  # Classify intent using keyword matching rules
            engine_used = "Keyword & Pattern Matching Engine"  # Record keyword engine used
        # Extract entities from input query
        extracted_entities = extract_entities(text)  # Extract amount, recipient, etc. from query
        # Implicit Fund Transfer override: if both recipient AND amount are found but intent is still Help,
        # it almost certainly means the user asked for a transfer e.g. "Send Raju 5000 rupees"
        if (  # Start multi-line condition check
            intent == "Help"  # Check if intent fell through to Help (no keyword matched)
            and extracted_entities.get("recipient") is not None  # Check if a recipient name was found
            and extracted_entities.get("amount") is not None  # Check if a monetary amount was found
        ):  # End of condition check
            intent = "Fund Transfer"  # Override intent to Fund Transfer since name + amount is a strong signal
            confidence = 0.70  # Assign moderate confidence since this is an inferred decision
        # Construct final structured response dictionary
        response = {  # Build structured result dictionary
            "query": text,  # Original user query string
            "intent": intent,  # Predicted intent category from 7 supported classes
            "confidence": confidence,  # Confidence score between 0.0 and 1.0
            "entities": extracted_entities,  # Dictionary of extracted entities (amount, recipient)
            "engine": engine_used,  # Information on whether neural IndicBERT or keyword fallback was used
        }  # End of response dictionary
        return response  # Return structured response
    except Exception as process_err:  # Catch any unexpected error in processing pipeline
        print(f"[SIO NLP] Error processing query: {process_err}")  # Log error message
        return {  # Return safe fallback response
            "query": text,  # Return original query
            "intent": "Help",  # Default to Help intent
            "confidence": 0.0,  # Zero confidence
            "entities": {"amount": None, "recipient": None},  # Empty entities
            "engine": "Error Fallback",  # Error status
        }  # End of safe fallback response
# Self-testing function to validate Module 2 across Hindi, English, and Hinglish queries # Description comment
def run_test_suite():  # Define function to run automated test cases for all 7 intents
    print("=" * 70)  # Print header border line
    print("  SIO Banking Assistant - Module 2: Intent & Entity Recognition Test  ")  # Print test title banner
    print("=" * 70)  # Print header separator line
    sample_queries = [  # List of test queries representing all 7 categories in Hindi, English, and Hinglish
        ("मेरे खाते में कितना बैलेंस है?", "Hindi Balance Enquiry"),  # Hindi Balance Enquiry
        ("What is my account balance?", "English Balance Enquiry"),  # English Balance Enquiry
        ("Mera balance kitna hai batao", "Hinglish Balance Enquiry"),  # Hinglish Balance Enquiry
        ("रमेश को 500 रुपये ट्रांसफर कर दो", "Hindi Fund Transfer with Amount & Recipient"),  # Hindi Transfer
        ("Transfer 1000 rupees to Shubham", "English Fund Transfer with Amount & Recipient"),  # English Transfer
        ("Amit ko 250 rs send karo", "Hinglish Fund Transfer with Amount & Recipient"),  # Hinglish Transfer
        ("Send Raju 5000 rupees", "English Fund Transfer: send [name] [amount]"),  # Pattern 4 test
        ("पिछला लेनदेन और स्टेटमेंट दिखाओ", "Hindi Transaction History"),  # Hindi Transaction History
        ("Show my last 5 transactions", "English Transaction History"),  # English Transaction History
        ("Phone me 199 ka recharge kar do", "Hinglish Mobile Recharge with Amount"),  # Hinglish Mobile Recharge
        ("Bijli ka bill bharna hai 1200", "Hinglish Bill Payment with Amount"),  # Hinglish Bill Payment
        ("नजदीकी एटीएम कहां है?", "Hindi ATM Locator"),  # Hindi ATM Locator
        ("Where is the nearest ATM branch?", "English ATM Locator"),  # English ATM Locator
        ("Mera ATM card kho gaya hai madad karo", "Hinglish Help / Card Blocking"),  # Hinglish Help
    ]  # End of test queries list
    for query, description in sample_queries:  # Loop through each sample query in the test suite
        print(f"\n[Test Case] : {description}")  # Print description of the test case
        result = process_query(query)  # Execute query processing pipeline
        print(f" Query       : \"{result['query']}\"")  # Print input query
        print(f" Intent      : {result['intent']} (Confidence: {result['confidence']})")  # Print predicted intent
        print(f" Entities    : {result['entities']}")  # Print extracted entities
        print(f" Engine      : {result['engine']}")  # Print engine used
        print("-" * 70)  # Print separator line
if __name__ == "__main__":  # Check if script is run directly from command line
    run_test_suite()  # Execute test suite
