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

# Define the core banking intent categories supported by SIO                     # Description comment
INTENT_CATEGORIES = [  # List of all 7 predefined banking intent categories
    "Balance Enquiry",  # Intent for checking current bank account balance
    "Fund Transfer",  # Intent for sending money to another person or account
    "Transaction History",  # Intent for viewing past account statements or transactions
    "Mobile Recharge",  # Intent for recharging prepaid phone numbers or data packs
    "Bill Payment",  # Intent for paying electricity, water, gas, or utility bills
    "ATM Locator",  # Intent for finding nearby ATMs or bank branches
    "Help",  # Intent for explicit customer care support or card blocking
    "Unknown",  # Fallback intent when user input does not match any category
]  # End of INTENT_CATEGORIES list

# Name of the IndicBERT pretrained model from HuggingFace ai4bharat              # Description comment
INDICBERT_MODEL_NAME = "ai4bharat/indic-bert"  # HuggingFace model identifier for IndicBERT

# Global cache variable for loaded IndicBERT model and tokenizer                 # Description comment
_MODEL_CACHE: Optional[Tuple[Optional[Any], Optional[Any]]] = None  # Cache model to avoid repeated lookups

# Comprehensive keyword dictionary for all banking intents                      # Description comment
KEYWORD_DICTIONARY = {  # Dictionary mapping each intent category to multilingual keywords
    "Balance Enquiry": [  # Keywords and phrases indicating a balance enquiry query
        "balance", "account balance", "check balance", "how much money", "funds", "remaining",  # English keywords
        "बैलेंस", "बेलेंस", "बैलेन्स", "बेलेन्स", "बैलंस", "खाता", "खाते में", "कितने पैसे",  # Hindi keywords and Whisper phonetic variations
        "रुपये हैं", "रकम", "शेष राशि", "राशि", "कितना है", "कितने पैसे हैं", "पैसे कितने हैं",  # Hindi balance status phrases
        "बैलेंस बताओ", "बेलेंस बताओ", "बैलेंस चेक", "बैलेंस देखना", "पैसे देखने", "खाता चेक",  # Common Hindi requests
        "बेलेंस कितना", "बैलेंस कितना", "कितना बैलेंस", "कितना बेलेंस", "कितना पैसा है", "बाकी पैसे",  # Common spoken Hindi combinations
        "balance kitna", "paise kitne", "account me kitna", "paisa dekhna", "kitne rupaye",  # Hinglish keywords
        "kitna paisa", "balance check", "khata balance", "balance batao", "paise batao",  # Additional Hinglish
    ],  # End of Balance Enquiry keyword list
    "Fund Transfer": [  # Keywords and phrases indicating a money transfer query
        "transfer", "send money", "pay to", "send cash", "remit", "send payment", "deposit",  # English keywords
        "send", "pay", "payment",  # Short English transfer action verbs
        "ट्रांसफर", "भेजें", "भेजना", "भेजो", "भेजने", "भेजने हैं", "भेजना है", "डालना", "डालो",  # Hindi verbs
        "पैसे डालना", "रुपये ट्रांसफर", "पैसे भेजो", "भुगतान करो", "पैसे भेजने हैं", "सेंड",  # Hindi phrases
        "पैसे ट्रांसफर", "रुपये भेजने", "पैसे डालने", "पैसे सेंड", "भेजना चाहता हूँ",  # Common spoken Hindi
        "transfer karo", "bhejna", "send karo", "transfer karna", "daal do", "bhej do", "pay karna",  # Hinglish
        "bhejo", "paise bhejo", "rupaye bhejo", "transfer kar do", "send kar do", "paise send",  # Hinglish phrases
    ],  # End of Fund Transfer keyword list
    "Transaction History": [  # Keywords and phrases indicating a transaction statement query
        "transaction", "history", "statement", "passbook", "recent payments", "mini statement",  # English keywords
        "लेनदेन", "इतिहास", "स्टेटमेंट", "पासबुक", "पिछला लेनदेन", "खर्च", "विवरण",  # Hindi keywords
        "history dikhao", "statement chahiye", "passbook update", "pichle transaction", "mini statement",  # Hinglish
        "last transaction", "kharch dekhna", "kahan kharch hua", "purane transaction",  # Additional Hinglish
    ],  # End of Transaction History keyword list
    "Mobile Recharge": [  # Keywords and phrases indicating a mobile recharge query
        "recharge", "top up", "mobile pack", "phone recharge", "data pack", "prepaid recharge",  # English keywords
        "रिचार्ज", "मोबाइल रिचार्ज", "फोन रिचार्ज", "टॉकटाइम", "मोबाइल", "फोन",  # Hindi keywords
        "mobile recharge", "phone me recharge", "recharge kar do", "number recharge", "recharge karna",  # Hinglish
        "sim recharge", "recharge plan", "topup", "talktime",  # Additional Hinglish phrases
    ],  # End of Mobile Recharge keyword list
    "Bill Payment": [  # Keywords and phrases indicating a utility bill payment query
        "bill", "electricity bill", "water bill", "gas bill", "utility", "pay bill", "power bill",  # English keywords
        "बिल", "बिजली का बिल", "पानी का बिल", "गैस बिल", "बिल भरना", "बिजली बिल", "पानी बिल",  # Hindi keywords
        "bijli bill", "water bill", "gas bill", "bill bharna", "bill pay", "light bill",  # Hinglish keywords
        "bill bhardo", "bill payment", "bijli ka bill", "current bill",  # Additional Hinglish phrases
    ],  # End of Bill Payment keyword list
    "ATM Locator": [  # Keywords and phrases indicating a search for an ATM or branch
        "nearest atm", "find atm", "where is atm", "bank branch", "atm location", "nearby atm",  # English keywords
        "closest atm", "cash withdrawal machine", "atm locator", "locate atm", "branch",  # English keywords
        "शाखा", "नजदीकी एटीएम", "पैसे निकालने की मशीन", "एटीएम कहां है", "बैंक शाखा", "एटीएम",  # Hindi keywords
        "paas ka atm", "atm kahan hai", "nearest branch", "atm dhoondo", "atm kidhar hai",  # Hinglish keywords
    ],  # End of ATM Locator keyword list
    "Help": [  # Keywords and phrases for explicit customer support or lost card reporting
        "customer care", "customer support", "agent", "executive", "lost card", "block card",  # English keywords
        "card lost", "lost my card", "stolen card", "card blocked", "report fraud", "complaint",  # English keywords
        "कस्टमर केयर", "कार्ड खो गया", "कार्ड ब्लॉक", "शिकायत", "धोखाधड़ी", "कार्ड बंद करो",  # Hindi keywords
        "खो गया", "सहायता केंद्र", "कस्टमर सर्विस", "एजेंट से बात", "अधिकारी से बात",  # Hindi keywords
        "customer care number", "card block karo", "card kho gaya", "agent se baat",  # Hinglish keywords
    ],  # End of Help keyword list
}  # End of KEYWORD_DICTIONARY

# Helper function to remove leading greetings from queries                        # Description comment
def clean_greetings(text: str) -> str:  # Define function to strip conversational greetings
    pattern = r'^(?:namaste|namaskar|hello|hi|hey|haan|namaste ji|नमस्ते|नमस्कार|हेलो|हाय)\s*,?\s*'  # Greeting regex
    return re.sub(pattern, '', text.strip(), flags=re.IGNORECASE).strip()  # Return cleaned text

# Function to classify intent using keyword and pattern matching                 # Description comment
def classify_intent_by_keywords(text: str) -> Tuple[str, float]:  # Define keyword-based intent classification function
    try:  # Start try block to handle string processing safely
        if not text or not isinstance(text, str):  # Check if input text is empty or invalid
            return "Unknown", 0.0  # Return Unknown intent with zero confidence if input is invalid
        cleaned_text = clean_greetings(text.lower().strip())  # Remove greetings and convert to lowercase
        if not cleaned_text:  # If nothing remains after removing greetings
            return "Unknown", 0.0  # Return Unknown intent
        scores = {intent: 0 for intent in KEYWORD_DICTIONARY.keys()}  # Initialize scores
        # Check ATM intent specifically if 'atm' word is present
        if "atm" in cleaned_text and not any(term in cleaned_text for term in ["block", "lost", "stolen", "kho gaya"]):  # Check ATM
            scores["ATM Locator"] += 2  # Give strong boost to ATM locator
        for intent, keywords in KEYWORD_DICTIONARY.items():  # Iterate through intents
            for keyword in keywords:  # Iterate through keywords
                keyword_clean = keyword.lower().strip()  # Clean keyword
                if keyword_clean in cleaned_text:  # Check substring match
                    weight = len(keyword_clean.split())  # Multi-word match gets higher weight
                    scores[intent] += weight  # Add score
        best_intent = max(scores, key=scores.get)  # Find intent with highest score
        highest_score = scores[best_intent]  # Get highest score value
        if highest_score == 0:  # Check if no keywords matched
            return "Unknown", 0.0  # Return Unknown intent instead of blindly defaulting to Help
        confidence = min(1.0, 0.50 + (highest_score * 0.15))  # Calculate confidence score
        return best_intent, round(confidence, 2)  # Return predicted intent and confidence
    except Exception as class_err:  # Catch any error during classification
        print(f"[SIO NLP] Error classifying intent: {class_err}")  # Log error
        return "Unknown", 0.0  # Return safe fallback

# Function to extract financial and user entities from input text                # Description comment
def extract_entities(text: str) -> Dict[str, Optional[Any]]:  # Define entity extraction function
    entities = {  # Initialize dictionary to store extracted entity values
        "amount": None,  # Extracted monetary transaction amount
        "recipient": None,  # Extracted recipient beneficiary name
    }  # End of entities dictionary
    try:  # Start try block to handle entity extraction safely
        if not text or not isinstance(text, str):  # Check if input text is empty or invalid
            return entities  # Return empty entities dictionary if text is not valid
        stopwords = {  # Words to filter out from recipient names
            "bank", "account", "paise", "paisa", "rupees", "khata", "mujhe", "kisi", "card",  # Noise words
            "rs", "inr", "the", "my", "please", "karo", "karna", "dena", "send", "pay", "bhejo",  # Noise words
            "बैलेंस", "बेलेंस", "बैलेन्स", "बेलेन्स", "खाता", "पैसे", "रुपये", "एटीएम", "मदद", "बिल", "फोन", "नंबर", "ko", "को",  # Hindi noise words
            "भेजना", "भेजने", "डालना", "डालो", "सेंड", "नमस्ते", "हेलो", "कितना", "कितने", "बताओ",  # Common action verbs and interrogatives
        }  # End of stopwords set
        # Pattern 1: Monetary amount (₹, Rs, INR, रुपये, or numbers)
        is_transaction_count = bool(re.search(r'\b(?:last|past|previous|पिछला|पिछले|pichle)\s+\d+', text, flags=re.IGNORECASE))  # Check count
        if not is_transaction_count:  # Only extract amount if not count
            amount_pattern = r'(?:₹|rs\.?|inr|रुपये|रुपए)?\s*(\b\d+(?:,\d+)*(?:\.\d+)?\b)\s*(?:₹|rs\.?|inr|रुपये|रुपए|rupees|rupee)?'  # Regex
            amount_matches = re.findall(amount_pattern, text, flags=re.IGNORECASE)  # Find matches
            if amount_matches:  # If amount matches found
                cleaned_amount = amount_matches[0].replace(",", "")  # Strip commas
                entities["amount"] = float(cleaned_amount) if "." in cleaned_amount else int(cleaned_amount)  # Parse
        # Pattern 2: Recipient in Hindi/Hinglish (e.g. "रमेश को", "Raju ko")
        recipient_ko_pattern = r'([A-Za-z\u0900-\u097F]+)\s+(?:को|ko)(?:\s+|$)'  # Recipient regex
        match_ko = re.search(recipient_ko_pattern, text, flags=re.IGNORECASE)  # Search
        if match_ko:  # If match found
            candidate = match_ko.group(1).strip()  # Extract candidate name
            if candidate.lower() not in stopwords:  # Filter stopwords
                entities["recipient"] = candidate  # Set recipient
        # Pattern 3: Recipient in English (e.g. "to Ramesh", "send to Raju")
        if not entities["recipient"]:  # If recipient not yet found
            recipient_en_pattern = r'(?:to|pay|transfer\s+to|send\s+to)\s+([A-Za-z\u0900-\u097F]+)'  # English regex
            match_en = re.search(recipient_en_pattern, text, flags=re.IGNORECASE)  # Search
            if match_en:  # If match found
                candidate = match_en.group(1).strip()  # Extract candidate name
                if candidate.lower() not in stopwords:  # Filter stopwords
                    entities["recipient"] = candidate  # Set recipient
        # Pattern 4: Recipient followed by amount (e.g. "send Raju 5000", "bhejo Raju 5000")
        if not entities["recipient"]:  # If recipient not yet found
            send_name_pattern = r'(?:send|pay|bhejo|transfer|डालो)\s+([A-Za-z\u0900-\u097F]+)\s+\d'  # Verb-name-number regex
            match_send = re.search(send_name_pattern, text, flags=re.IGNORECASE)  # Search
            if match_send:  # If match found
                candidate = match_send.group(1).strip()  # Extract candidate name
                if candidate.lower() not in stopwords:  # Filter stopwords
                    entities["recipient"] = candidate  # Set recipient
    except Exception as ent_err:  # Catch any error
        print(f"[SIO NLP] Error extracting entities: {ent_err}")  # Log error
    return entities  # Return entities dict

# Main analysis function combining model inference and entity extraction         # Description comment
def process_query(text: str) -> Dict[str, Any]:  # Define primary entry point for NLP Module 2
    try:  # Start try block for end-to-end processing
        if not text or not isinstance(text, str):  # Check validity
            text = ""  # Fallback to empty string
        intent, confidence = classify_intent_by_keywords(text)  # Classify intent
        extracted_entities = extract_entities(text)  # Extract entities
        # Override to Fund Transfer if recipient and amount are both present
        if intent == "Unknown" and extracted_entities.get("recipient") and extracted_entities.get("amount"):  # Check
            intent = "Fund Transfer"  # Override intent
            confidence = 0.70  # Assign confidence
        return {  # Return structured response dictionary
            "query": text,  # Query text
            "intent": intent,  # Predicted intent
            "confidence": confidence,  # Confidence score
            "entities": extracted_entities,  # Extracted entities
            "engine": "Keyword & Pattern Matching Engine",  # Engine identifier
        }  # End of dictionary
    except Exception as err:  # Catch any processing error
        print(f"[SIO NLP] Error processing query: {err}")  # Log error
        return {  # Safe fallback
            "query": text, "intent": "Unknown", "confidence": 0.0,  # Fallback values
            "entities": {"amount": None, "recipient": None}, "engine": "Error Fallback"  # Fallback
        }  # End of fallback dictionary
