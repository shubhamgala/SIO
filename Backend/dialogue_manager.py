# SIO Voice-First Banking Assistant - Module 3: Dialogue Manager                  # File title comment
# This module sits between Module 2 (NLP) and future Module 4 (Banking Logic).  # Module position comment
# It checks if all required info is collected, asks follow-up voice questions,   # Responsibility comment
# reads back transaction details in Hindi, and waits for voice confirmation.     # Responsibility comment

import os  # Import os module for file path and environment operations
import sys  # Import sys module for runtime configuration and path management
import re  # Import re module for regular expression text matching
import time  # Import time module for small pauses between voice prompts
import tempfile  # Import tempfile module to create secure temporary audio files
import subprocess  # Import subprocess module to install missing packages via pip

# Reconfigure stdout and stderr to UTF-8 so Hindi characters print correctly     # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if stdout supports encoding reconfiguration
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout to UTF-8 so Hindi prints without errors
if hasattr(sys.stderr, 'reconfigure'):  # Check if stderr supports encoding reconfiguration
    sys.stderr.reconfigure(encoding='utf-8')  # Set stderr to UTF-8 so error messages print correctly

# Add project root to sys.path so STT and NLP modules can be imported cleanly   # Description comment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add project root folder to path

# ─── Dependency installer ────────────────────────────────────────────────────── #
# Function to install a pip package if it is not already available               # Description comment
def _install_package(package_name):  # Define helper function to auto-install a missing package
    try:  # Start try block to test if the package is importable
        __import__(package_name.replace('-', '_'))  # Try importing the package using its import name
    except ImportError:  # Catch ImportError if the package is not installed
        print(f"[SIO DM] Installing missing package '{package_name}'...")  # Inform user about auto-install
        try:  # Start try block to run pip install via subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])  # Run pip install
            print(f"[SIO DM] Package '{package_name}' installed successfully.")  # Confirm successful install
        except Exception as pip_err:  # Catch any error during pip installation
            print(f"[SIO DM] Warning: Could not install '{package_name}': {pip_err}")  # Print warning message

# Ensure audio playback and TTS libraries are installed before importing them    # Description comment
_install_package('gtts')  # Install Google Text-to-Speech library if missing
_install_package('pygame')  # Install pygame for cross-platform audio playback

# ─── Imports ─────────────────────────────────────────────────────────────────── #
try:  # Start try block to import gTTS for Hindi text-to-speech conversion
    from gtts import gTTS  # Import gTTS class to generate Hindi speech audio files
    GTTS_AVAILABLE = True  # Mark gTTS as available for use
except ImportError:  # Catch ImportError if gTTS could not be imported
    GTTS_AVAILABLE = False  # Mark gTTS as unavailable, will fall back to print
    print('[SIO DM] Warning: gTTS not available. Voice output will be text only.')  # Warn user

try:  # Start try block to import pygame for audio playback
    import pygame  # Import pygame for playing generated speech audio files
    pygame.mixer.init()  # Initialize the pygame audio mixer subsystem
    PYGAME_AVAILABLE = True  # Mark pygame as available for use
except Exception as pg_err:  # Catch any error during pygame import or initialization
    PYGAME_AVAILABLE = False  # Mark pygame as unavailable, will fall back to print
    print(f'[SIO DM] Warning: pygame not available: {pg_err}. Audio playback disabled.')  # Warn user

# ─── Intent slot requirements ─────────────────────────────────────────────────── #
# Dictionary defining which entity slots are required for each banking intent    # Description comment
# If a slot is missing the dialogue manager will ask a follow-up voice question  # Description comment
REQUIRED_SLOTS = {  # Map each intent to the list of entity fields it needs before proceeding
    'Fund Transfer':      ['recipient', 'amount'],  # Transfer needs both who to send to and how much
    'Mobile Recharge':    ['amount'],  # Recharge only needs the amount (phone number assumed registered)
    'Bill Payment':       ['bill_type'],  # Bill payment needs which bill type (electricity, water, gas)
    'Balance Enquiry':    [],  # Balance check needs no extra information
    'Transaction History': [],  # History request needs no extra information
    'ATM Locator':        [],  # ATM search needs no extra information
    'Help':               [],  # Help request needs no extra information
}  # End of REQUIRED_SLOTS dictionary

# ─── Follow-up question templates ─────────────────────────────────────────────── #
# Hindi follow-up questions asked when a required entity slot is missing          # Description comment
FOLLOWUP_QUESTIONS = {  # Map each missing slot to the Hindi voice question to ask
    'recipient': 'Aap kisko paise bhejna chahte hain?',  # Ask who to send money to
    'amount':    'Aap kitne rupaye bhejna chahte hain?',  # Ask how much money to send
    'bill_type': 'Kaun sa bill bharna hai? Bijli, paani, ya gas?',  # Ask which bill type
}  # End of FOLLOWUP_QUESTIONS dictionary

# ─── Confirmation word sets ────────────────────────────────────────────────────── #
# Words in Hindi, Hinglish and English that mean YES — confirm the transaction   # Description comment
YES_WORDS = {  # Set of words across languages that indicate user confirmation
    'haan', 'ha', 'han', 'yes', 'yeah', 'yep', 'yup', 'correct',  # Hindi, Hinglish, English yes words
    'theek hai', 'theek', 'bilkul', 'zaroor', 'confirm', 'ok', 'okay',  # Additional agreement words
    'sahi', 'proceed', 'go ahead', 'kar do', 'bhej do', 'kardo', 'bhejdo',  # Action confirmation words
    'ah', 'aah', 'haa', 'ji haan', 'haan ji', 'haanji', 'ji', 'jee',  # Whisper phonetics and polite Hindi terms
    'हाँ', 'हां', 'हं', 'जी हाँ', 'जी हां', 'सही है', 'कर दो', 'भेज दो',  # Devanagari script affirmatives
}  # End of YES_WORDS set

# Words in Hindi, Hinglish and English that mean NO — cancel the transaction     # Description comment
NO_WORDS = {  # Set of words across languages that indicate user cancellation
    'nahi', 'na', 'no', 'nope', 'cancel', 'naa', 'mat karo',  # Hindi, Hinglish, English no words
    'rok do', 'band karo', 'stop', 'abort', 'mat', 'nahin',  # Additional cancellation words
    'nah', 'nhi', 'nahi chahiye', 'matkaro', 'rokdo',  # Common phonetic variations
    'नहीं', 'ना', 'मत', 'मत करो', 'रोक दो', 'बंद करो', 'रद्द करो',  # Devanagari script cancellation words
}  # End of NO_WORDS set

# ─── Hindi number to words converter (for natural speech readback) ───────────── #
# Dictionary mapping numbers 0 to 99 to their Hindi Devanagari words              # Description comment
HINDI_0_TO_99 = {  # Map numbers 0 to 99 to their authentic spoken Hindi words
    0: 'शून्य', 1: 'एक', 2: 'दो', 3: 'तीन', 4: 'चार', 5: 'पाँच',  # Numbers 0-5
    6: 'छह', 7: 'सात', 8: 'आठ', 9: 'नौ', 10: 'दस',  # Numbers 6-10
    11: 'ग्यारह', 12: 'बारह', 13: 'तेरह', 14: 'चौदह', 15: 'पंद्रह',  # Numbers 11-15
    16: 'सोलह', 17: 'सत्रह', 18: 'अठारह', 19: 'उन्नीस', 20: 'बीस',  # Numbers 16-20
    21: 'इक्कीस', 22: 'बाईस', 23: 'तेईस', 24: 'चौबीस', 25: 'पच्चीस',  # Numbers 21-25
    26: 'छब्बीस', 27: 'सत्ताईस', 28: 'अट्ठाईस', 29: 'उनतीस', 30: 'तीस',  # Numbers 26-30
    31: 'इकतीस', 32: 'बत्तीस', 33: 'तैंतीस', 34: 'चौंतीस', 35: 'पैंतीस',  # Numbers 31-35
    36: 'छत्तीस', 37: 'सैंतीस', 38: 'अड़तीस', 39: 'उनतालीस', 40: 'चालीस',  # Numbers 36-40
    41: 'इकतालीस', 42: 'बयालीस', 43: 'तैंतालीस', 44: 'चवालीस', 45: 'पैंतालीस',  # Numbers 41-45
    46: 'छियालीस', 47: 'सैंतालीस', 48: 'अड़तालीस', 49: 'उनचास', 50: 'पचास',  # Numbers 46-50
    51: 'इक्यावन', 52: 'बावन', 53: 'तिरेपन', 54: 'चौवन', 55: 'पचपन',  # Numbers 51-55
    56: 'छप्पन', 57: 'सत्तावन', 58: 'अट्ठावन', 59: 'उनसठ', 60: 'साठ',  # Numbers 56-60
    61: 'इकसठ', 62: 'बासठ', 63: 'तिरेसठ', 64: 'चौंसठ', 65: 'पैंसठ',  # Numbers 61-65
    66: 'छियासठ', 67: 'सरसठ', 68: 'अड़सठ', 69: 'उनहत्तर', 70: 'सत्तर',  # Numbers 66-70
    71: 'इकहत्तर', 72: 'बहत्तर', 73: 'तिहत्तर', 74: 'चौहत्तर', 75: 'पचहत्तर',  # Numbers 71-75
    76: 'छिहत्तर', 77: 'सतहत्तर', 78: 'अठहत्तर', 79: 'उनासी', 80: 'अस्सी',  # Numbers 76-80
    81: 'इक्यासी', 82: 'बयासी', 83: 'तिरासी', 84: 'चौरासी', 85: 'पचासी',  # Numbers 81-85
    86: 'छियासी', 87: 'सत्तासी', 88: 'अट्ठासी', 89: 'नवासी', 90: 'नब्बे',  # Numbers 86-90
    91: 'इक्यानवे', 92: 'बानवे', 93: 'तिरानवे', 94: 'चौरानवे', 95: 'पंचानवे',  # Numbers 91-95
    96: 'छियानवे', 97: 'सत्तानवे', 98: 'अट्ठानवे', 99: 'निन्यानवे',  # Numbers 96-99
}  # End of HINDI_0_TO_99 dictionary

# Function to convert an integer into spoken Hindi words                         # Description comment
def int_to_hindi(num):  # Define function to convert integer into Hindi word representation
    if num == 0:  # Check if number is zero
        return HINDI_0_TO_99[0]  # Return Hindi word for zero
    parts = []  # Initialize empty list to accumulate word segments
    crore = num // 10000000  # Calculate crore segment (10 million)
    if crore > 0:  # Check if crore segment is non-zero
        parts.append(int_to_hindi(crore) + ' करोड़')  # Append crore words
        num %= 10000000  # Remove crore portion from number
    lakh = num // 100000  # Calculate lakh segment (100 thousand)
    if lakh > 0:  # Check if lakh segment is non-zero
        parts.append(HINDI_0_TO_99.get(lakh, str(lakh)) + ' लाख')  # Append lakh words
        num %= 100000  # Remove lakh portion from number
    thousand = num // 1000  # Calculate thousand segment
    if thousand > 0:  # Check if thousand segment is non-zero
        parts.append(HINDI_0_TO_99.get(thousand, str(thousand)) + ' हज़ार')  # Append thousand words
        num %= 1000  # Remove thousand portion from number
    hundred = num // 100  # Calculate hundred segment
    if hundred > 0:  # Check if hundred segment is non-zero
        parts.append(HINDI_0_TO_99.get(hundred, str(hundred)) + ' सौ')  # Append hundred words
        num %= 100  # Remove hundred portion from number
    if num > 0:  # Check if remainder between 1 and 99 remains
        parts.append(HINDI_0_TO_99.get(num, str(num)))  # Append remainder Hindi words
    return ' '.join(parts)  # Return full combined Hindi word string

# Function to replace all numeric digits with authentic Hindi words for speech   # Description comment
def format_numbers_for_hindi_speech(text):  # Define function to prepare text with Hindi numbers for TTS
    def _replacer(match):  # Inner function to replace each matched numeric string
        raw = match.group(0).replace(',', '')  # Strip commas from number format
        try:  # Start try block to parse number safely
            val = float(raw)  # Convert raw numeric string to float
            if val >= 10000000:  # Ignore phone numbers and account numbers over 7 digits
                return match.group(0)  # Return original string for large account or phone numbers
            int_val = int(val)  # Extract integer portion
            decimal_part = int(round((val - int_val) * 100))  # Extract 2-decimal paise portion
            words = int_to_hindi(int_val)  # Convert integer portion into Hindi words
            if decimal_part > 0:  # Check if non-zero decimal paise exist
                words += ' दशमलव ' + int_to_hindi(decimal_part)  # Append decimal Hindi words
            return words  # Return spoken Hindi word replacement
        except Exception:  # Catch any parsing exception
            return match.group(0)  # Fall back to original match on error
    return re.sub(r'\b\d[\d,]*(\.\d+)?\b', _replacer, text)  # Replace all numbers in text

# ─── Voice output ─────────────────────────────────────────────────────────────── #
# Function to speak a Hindi text string aloud using gTTS and pygame              # Description comment
def speak(text):  # Define function to convert Hindi text to speech and play it
    print(f'[SIO DM] 🔊 {text}')  # Always print the text so it is visible even without audio
    if not GTTS_AVAILABLE or not PYGAME_AVAILABLE:  # Check if audio output libraries are available
        return  # Exit early and rely on printed text if audio is not available
    tmp_path = None  # Initialize temporary file path variable to None
    try:  # Start try block for TTS generation and audio playback
        spoken_text = format_numbers_for_hindi_speech(text)  # Convert digits (e.g. 500) to Hindi words ('पाँच सौ')
        tts = gTTS(text=spoken_text, lang='hi', slow=False)  # Create Hindi TTS audio object with authentic Hindi words
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:  # Create temp MP3 file
            tmp_path = tmp.name  # Store the temporary file path for later playback and cleanup
        tts.save(tmp_path)  # Save generated speech audio to the temporary MP3 file
        pygame.mixer.music.load(tmp_path)  # Load the saved MP3 file into pygame audio mixer
        pygame.mixer.music.play()  # Start playing the audio through the system speakers
        while pygame.mixer.music.get_busy():  # Loop while audio is still playing
            time.sleep(0.1)  # Wait 100ms before checking again to avoid busy-waiting
        pygame.mixer.music.unload()  # Unload the audio file from memory after playback
    except Exception as tts_err:  # Catch any error during TTS generation or playback
        print(f'[SIO DM] Warning: Could not play voice: {tts_err}')  # Print warning if audio fails
    finally:  # Always clean up the temporary MP3 file regardless of success or failure
        if tmp_path and os.path.exists(tmp_path):  # Check if the temporary file exists on disk
            try:  # Start try block for safe file deletion
                os.remove(tmp_path)  # Delete the temporary MP3 audio file from disk
            except OSError:  # Catch any OS error during file deletion
                pass  # Silently ignore file cleanup errors

# ─── Voice input (for follow-up questions) ────────────────────────────────────── #
# Function to record a short voice response from the user via the microphone     # Description comment
def listen_for_response(whisper_model, duration=5):  # Define function to record and transcribe short answer
    try:  # Start try block for voice capture
        from STT.whisper_stt import record_audio_to_wav, transcribe_audio, remove_temporary_file  # Import STT tools
        wav_path = record_audio_to_wav()  # Record audio from the microphone (with 3-second countdown)
        if not wav_path:  # Check if recording failed or microphone is unavailable
            print('[SIO DM] Warning: Could not record audio for follow-up response.', flush=True)  # Warn user
            return None  # Return None to signal that no response was captured
        try:  # Start inner try block to transcribe and clean up
            text, lang = transcribe_audio(whisper_model, wav_path)  # Transcribe the recorded response
            return text.strip() if text else None  # Return cleaned text or None if transcription was empty
        finally:  # Ensure temporary audio file is always cleaned up
            remove_temporary_file(wav_path)  # Delete the temporary WAV file from disk
    except Exception as listen_err:  # Catch any unexpected error during voice capture
        print(f'[SIO DM] Error during voice input: {listen_err}', flush=True)  # Print error message
        return None  # Return None so the dialogue can handle missing input gracefully

# ─── Confirmation-specific voice input ───────────────────────────────────────── #
# Dedicated function for recording YES/NO confirmations only (shorter, no echo)  # Description comment
def listen_for_confirmation(whisper_model):  # Define specialized confirmation listener
    try:  # Start try block
        from STT.whisper_stt import record_audio_short, transcribe_confirmation, remove_temporary_file  # Import specialized tools
        print('[SIO DM] Listening for Haan / Nahi...', flush=True)  # Inform user that confirmation is expected
        wav_path = record_audio_short(duration=3)  # Record only 3 seconds (echo wait is inside record_audio_short)
        if not wav_path:  # Check if recording failed
            print('[SIO DM] Warning: Could not record confirmation.', flush=True)  # Warn user
            return None  # Return None on failure
        try:  # Start inner try block to transcribe
            text, lang = transcribe_confirmation(whisper_model, wav_path)  # Transcribe forced to Hindi
            print(f'[SIO DM] Confirmation heard: "{text}" (lang={lang})', flush=True)  # Log what was heard
            return text.strip() if text else None  # Return cleaned text or None
        finally:  # Always clean up temp file
            remove_temporary_file(wav_path)  # Delete WAV file from disk
    except Exception as conf_err:  # Catch any error
        print(f'[SIO DM] Error during confirmation listen: {conf_err}', flush=True)  # Log error
        return None  # Return None gracefully


# ─── Yes / No detection ───────────────────────────────────────────────────────── #
# Function to interpret a spoken response as YES, NO, or UNKNOWN                # Description comment
def detect_yes_no(text):  # Define function to classify a voice response as yes, no, or unknown
    if not text:  # Check if the text is None or empty string
        return 'unknown'  # Return unknown if no text was captured
    lower = text.lower().strip()  # Convert to lowercase and strip whitespace for reliable matching
    # Strip punctuation so that 'Ah!', 'Haan.', or 'Nahi?' normalize cleanly without symbols # Cleaning step
    cleaned = re.sub(r'[^\w\s\u0900-\u097F]', ' ', lower).strip()  # Remove punctuation preserving Hindi unicode
    tokens = set(cleaned.split())  # Split cleaned string into individual word tokens
    for phrase in YES_WORDS:  # Iterate through all known yes-words and phrases
        if ' ' in phrase and phrase in cleaned:  # Check multi-word phrase matching directly in text
            return 'yes'  # Return yes if matching confirmation phrase is found
        elif phrase in tokens or phrase == cleaned:  # Check single word in tokens or exact match
            return 'yes'  # Return yes if confirmation word is found
    for phrase in NO_WORDS:  # Iterate through all known no-words and phrases
        if ' ' in phrase and phrase in cleaned:  # Check multi-word cancellation phrase directly
            return 'no'  # Return no if matching cancellation phrase is found
        elif phrase in tokens or phrase == cleaned:  # Check single word in tokens or exact match
            return 'no'  # Return no if cancellation token is found
    return 'unknown'  # Return unknown if neither yes nor no words were found

# ─── Amount extractor for follow-up ──────────────────────────────────────────── #
# Function to extract a numeric amount from a spoken follow-up answer            # Description comment
def extract_amount_from_text(text):  # Define function to parse a number from free-form spoken text
    if not text:  # Check if the text is None or empty
        return None  # Return None if there is no text to parse
    text_lower = text.lower()  # Convert text to lowercase for case-insensitive matching
    # Replace common spoken Indian number words with their numeric equivalents   # Step comment
    word_map = {  # Dictionary of spoken number words to digit strings
        'ek sau': '100', 'do sau': '200', 'teen sau': '300', 'chaar sau': '400',  # Hindi hundreds
        'paanch sau': '500', 'chhe sau': '600', 'saat sau': '700', 'aath sau': '800',  # Hindi hundreds
        'nau sau': '900', 'ek hazaar': '1000', 'do hazaar': '2000', 'panch hazaar': '5000',  # Hindi thousands
        'das hazaar': '10000', 'ek lakh': '100000', 'do lakh': '200000',  # Hindi large numbers
    }  # End of word_map dictionary
    for word, digit in word_map.items():  # Iterate through each word-to-digit mapping
        text_lower = text_lower.replace(word, digit)  # Replace spoken word with its digit equivalent
    matches = re.findall(r'\b\d[\d,]*\b', text_lower)  # Find all numeric patterns in processed text
    if matches:  # Check if any numbers were found
        cleaned = matches[0].replace(',', '')  # Remove comma separators from the first matched number
        try:  # Start try block to safely convert string to integer
            return int(cleaned)  # Return the integer value of the extracted amount
        except ValueError:  # Catch error if string is not a valid integer
            return None  # Return None if conversion fails
    return None  # Return None if no numbers were found in the text

# ─── Bill type extractor ──────────────────────────────────────────────────────── #
# Function to detect which bill type the user mentions in their response         # Description comment
def extract_bill_type_from_text(text):  # Define function to identify bill type from spoken text
    if not text:  # Check if the text is None or empty
        return None  # Return None if there is nothing to analyse
    lower = text.lower()  # Convert text to lowercase for reliable keyword matching
    if any(w in lower for w in ['bijli', 'electricity', 'light', 'current', 'electric']):  # Check electricity
        return 'Electricity'  # Return electricity bill type if any electricity keyword is found
    if any(w in lower for w in ['paani', 'water', 'jal']):  # Check water bill keywords
        return 'Water'  # Return water bill type if any water keyword is found
    if any(w in lower for w in ['gas', 'lpg', 'cylinder']):  # Check gas bill keywords
        return 'Gas'  # Return gas bill type if any gas keyword is found
    if any(w in lower for w in ['phone', 'mobile', 'recharge', 'internet', 'broadband']):  # Check telecom
        return 'Phone'  # Return phone bill type if telecom keyword is found
    return None  # Return None if no bill type could be detected

# ─── Recipient extractor for follow-up ───────────────────────────────────────── #
# Function to extract a recipient name from a follow-up voice response           # Description comment
def extract_recipient_from_text(text):  # Define function to parse a recipient name from spoken text
    if not text:  # Check if text is None or empty
        return None  # Return None if there is no text to process
    # Noise words to ignore when extracting name from a follow-up answer         # Comment
    noise_words = {  # Set of common noise words that should not be treated as recipient names
        'ko', 'को', 'bhejna', 'bhejiye', 'send', 'transfer', 'paise', 'rupaye',  # Transfer noise words
        'chahta', 'chahti', 'hun', 'hoon', 'hai', 'mujhe', 'please', 'karo',  # Grammar noise words
    }  # End of noise_words set
    tokens = text.strip().split()  # Split the spoken response into individual word tokens
    for token in tokens:  # Iterate through each word token
        clean = re.sub(r'[^\w]', '', token)  # Remove any non-word characters from the token
        if len(clean) >= 2 and clean.lower() not in noise_words:  # Check minimum length and not noise
            return clean.capitalize()  # Return the cleaned token as the recipient name (capitalized)
    return None  # Return None if no suitable name token was found

# ─── Confirmation message builder ─────────────────────────────────────────────── #
# Function to build a Hindi confirmation readback message from collected slots   # Description comment
def build_confirmation_message(intent, entities):  # Define function to assemble Hindi confirmation text
    if intent == 'Fund Transfer':  # Check if the intent is a fund transfer
        recipient = entities.get('recipient', 'recipient')  # Get recipient name or fallback placeholder
        amount = entities.get('amount', 0)  # Get transaction amount or fallback to zero
        return f'Kya aap {recipient} ko {amount} rupaye bhejna chahte hain? Haan ya Nahi?'  # Build message
    if intent == 'Mobile Recharge':  # Check if the intent is a mobile recharge
        amount = entities.get('amount', 0)  # Get recharge amount or fallback to zero
        return f'Kya aap {amount} rupaye ka mobile recharge karna chahte hain? Haan ya Nahi?'  # Build message
    if intent == 'Bill Payment':  # Check if the intent is a bill payment
        bill_type = entities.get('bill_type', 'bill')  # Get bill type or fallback placeholder
        amount = entities.get('amount', '')  # Get amount if available for bill payment
        amount_part = f'{amount} rupaye ka ' if amount else ''  # Build optional amount string
        return f'Kya aap {amount_part}{bill_type} bill bharna chahte hain? Haan ya Nahi?'  # Build message
    if intent == 'Balance Enquiry':  # Check if the intent is a balance enquiry
        return 'Kya aap apna account balance check karna chahte hain? Haan ya Nahi?'  # Build message
    if intent == 'Transaction History':  # Check if the intent is a transaction history request
        return 'Kya aap apna transaction history dekhna chahte hain? Haan ya Nahi?'  # Build message
    if intent == 'ATM Locator':  # Check if the intent is an ATM locator request
        return 'Kya aap paas ka ATM dhundhna chahte hain? Haan ya Nahi?'  # Build message
    if intent == 'Help':  # Check if the intent is a help request
        return 'Kya aap customer care se baat karna chahte hain? Haan ya Nahi?'  # Build message
    return 'Kya aap yeh kaam karna chahte hain? Haan ya Nahi?'  # Generic fallback confirmation message

# ─── Core dialogue runner ─────────────────────────────────────────────────────── #
# Main entry point: given NLP result, run the full dialogue to collect and confirm # Description comment
def run_dialogue(nlp_result, whisper_model):  # Define the main dialogue loop function
    """
    Accepts the NLP result dictionary from Module 2 and the loaded Whisper model.
    Returns a result dictionary with keys:
        status   -> 'confirmed', 'cancelled', or 'error'
        intent   -> final intent string
        entities -> final collected entities dictionary
        message  -> human-readable Hindi status message
    """  # End of docstring
    try:  # Start outer try block to catch any unexpected dialogue errors
        intent = nlp_result.get('intent', 'Help')  # Extract classified intent from NLP result
        entities = dict(nlp_result.get('entities', {}))  # Copy entities dict so we can modify it safely
        confidence = nlp_result.get('confidence', 0.0)  # Extract confidence score from NLP result

        print('=' * 70)  # Print top banner separator
        print(f'[SIO DM] Intent detected : {intent} (confidence: {confidence})')  # Log detected intent
        print('=' * 70)  # Print banner separator

        # ── Step 0: Handle unrecognised or low-confidence intent ───────────── #
        if intent == 'Unknown' or confidence < 0.35:  # Check if intent was not recognized or confidence too low
            speak('Maafi kijiye, mujhe aapki baat samajh nahi aayi. Kripya dobara bolein.')  # Speak apology
            return {  # Return cancelled result safely
                'status': 'cancelled',  # Mark status as cancelled
                'intent': 'Unknown',  # Set intent as Unknown
                'entities': entities,  # Retain collected entities
                'message': 'Intent not recognized. Please speak your banking request clearly.',  # Message
            }  # End of return dictionary

        # ── Step 1: Slot filling — collect any missing required information ── #
        required = REQUIRED_SLOTS.get(intent, [])  # Retrieve list of required slots for this intent
        for slot in required:  # Iterate through each required slot for the detected intent
            if entities.get(slot):  # Check if this slot already has a value from Module 2
                continue  # Skip to next slot if this one is already filled
            # Slot is missing — ask the user a follow-up voice question             # Comment
            if slot == 'amount' and intent == 'Mobile Recharge':  # Check if asking amount for mobile recharge
                question = 'Aap kitne rupaye ka recharge karna chahte hain?'  # Ask specific recharge amount
            else:  # Standard question lookup
                question = FOLLOWUP_QUESTIONS.get(slot, f'Please provide your {slot}.')  # Get follow-up question
            print(f'\n[SIO DM] Missing slot: "{slot}" — asking follow-up question...')  # Log missing slot
            speak(question)  # Speak the follow-up question aloud in Hindi
            print('[SIO DM] Listening for your answer...')  # Inform user we are listening
            response_text = listen_for_response(whisper_model)  # Record and transcribe user's spoken answer
            if not response_text:  # Check if no voice response was captured
                speak('Maafi kijiye, mujhe aapki awaaz sunai nahi di. Kripya dobara bolein.')  # Apology
                response_text = listen_for_response(whisper_model)  # Give user one more attempt
            if not response_text:  # Check again if still no response after second attempt
                speak('Koi jawab nahi mila. Yeh kaam cancel kar rahe hain.')  # Announce cancellation
                return {  # Return cancellation result dictionary
                    'status': 'cancelled',  # Mark status as cancelled
                    'intent': intent,  # Include the detected intent
                    'entities': entities,  # Include whatever entities were collected so far
                    'message': 'No voice response received for follow-up question.',  # Error description
                }  # End of cancellation return
            print(f'[SIO DM] Follow-up response heard: "{response_text}"')  # Log transcribed follow-up
            # Extract the relevant entity value from the follow-up response       # Comment
            if slot == 'recipient':  # Check if we are filling the recipient slot
                extracted_value = extract_recipient_from_text(response_text)  # Extract name from answer
            elif slot == 'amount':  # Check if we are filling the amount slot
                extracted_value = extract_amount_from_text(response_text)  # Extract number from answer
            elif slot == 'bill_type':  # Check if we are filling the bill type slot
                extracted_value = extract_bill_type_from_text(response_text)  # Extract bill type
            else:  # For any unrecognized slot type
                extracted_value = response_text.strip()  # Use raw response text as fallback value
            if extracted_value:  # Check if a value was successfully extracted from the response
                entities[slot] = extracted_value  # Store the extracted value into the entities dictionary
                print(f'[SIO DM] Slot "{slot}" filled with: {extracted_value}')  # Log filled slot
            else:  # If extraction failed to find a usable value
                speak(f'Maafi kijiye, yeh samajh nahi aaya. Kripya dobara bolein.')  # Ask user to repeat
                second_response = listen_for_response(whisper_model)  # Record second attempt
                if slot == 'recipient':  # Check slot type for second extraction attempt
                    extracted_value = extract_recipient_from_text(second_response)  # Try extracting again
                elif slot == 'amount':  # Check slot type for second extraction attempt
                    extracted_value = extract_amount_from_text(second_response)  # Try extracting again
                elif slot == 'bill_type':  # Check slot type for second extraction attempt
                    extracted_value = extract_bill_type_from_text(second_response)  # Try extracting again
                else:  # For unrecognized slot on second attempt
                    extracted_value = second_response.strip() if second_response else None  # Use raw text
                if extracted_value:  # Check if second extraction attempt succeeded
                    entities[slot] = extracted_value  # Store the extracted value into entities
                    print(f'[SIO DM] Slot "{slot}" filled on second attempt: {extracted_value}')  # Log
                else:  # If still no value after two attempts
                    speak('Yeh jaankari nahi mili. Kaam cancel kar rahe hain.')  # Announce failure
                    return {  # Return cancellation result dictionary
                        'status': 'cancelled',  # Mark status as cancelled
                        'intent': intent,  # Include the detected intent
                        'entities': entities,  # Include what was collected
                        'message': f'Could not extract value for required slot: {slot}',  # Error message
                    }  # End of cancellation return

        # ── Step 2: Read back complete details and ask for confirmation ──────── #
        print('\n[SIO DM] All required slots filled. Building confirmation...', flush=True)  # Log slot completion
        confirmation_msg = build_confirmation_message(intent, entities)  # Build Hindi confirmation sentence
        print(f'[SIO DM] Confirmation prompt: {confirmation_msg}', flush=True)  # Log confirmation message text
        speak(confirmation_msg)  # Speak the full confirmation message aloud to the user

        # ── Step 3: Listen for Haan (yes) or Nahi (no) from the user ────────── #
        max_confirmation_attempts = 2  # Allow user up to 2 attempts to confirm or cancel
        for attempt in range(max_confirmation_attempts):  # Loop for up to 2 confirmation attempts
            print(f'[SIO DM] Listening for confirmation (attempt {attempt + 1})...', flush=True)  # Log attempt
            confirm_text = listen_for_confirmation(whisper_model)  # Use dedicated confirmation recorder (3s, Hindi forced, echo-safe)
            print(f'[SIO DM] Confirmation response heard: "{confirm_text}"', flush=True)  # Log what was heard
            decision = detect_yes_no(confirm_text)  # Classify response as yes, no, or unknown
            if decision == 'yes':  # Check if user confirmed the transaction
                speak('Theek hai! Aapka kaam ho raha hai.')  # Acknowledge and proceed with Hindi voice
                print('\n[SIO DM] ✅ User CONFIRMED. Proceeding to banking action.', flush=True)  # Log confirmation
                return {  # Return confirmed result dictionary
                    'status': 'confirmed',  # Mark status as confirmed
                    'intent': intent,  # Include the final intent
                    'entities': entities,  # Include all collected entities
                    'message': 'User confirmed. Ready for banking action.',  # Success message
                }  # End of confirmed return
            elif decision == 'no':  # Check if user cancelled the transaction
                speak('Theek hai, yeh kaam cancel kar diya gaya hai.')  # Acknowledge cancellation
                print('\n[SIO DM] ❌ User CANCELLED. Aborting transaction.', flush=True)  # Log cancellation
                return {  # Return cancelled result dictionary
                    'status': 'cancelled',  # Mark status as cancelled
                    'intent': intent,  # Include the intent that was cancelled
                    'entities': entities,  # Include entities that were collected
                    'message': 'User cancelled the transaction.',  # Cancellation message
                }  # End of cancelled return
            else:  # If the response was neither yes nor no
                if attempt < max_confirmation_attempts - 1:  # Check if we have more attempts left
                    speak('Kripya Haan ya Nahi bolein.')  # Ask user to say yes or no clearly
                else:  # If this was the final attempt and still unclear
                    speak('Koi jawab nahi mila. Kaam cancel kar rahe hain.')  # Announce timeout
        # Fell through all attempts without a clear yes or no                    # Comment

        return {  # Return cancelled result due to timeout
            'status': 'cancelled',  # Mark status as cancelled
            'intent': intent,  # Include the intent
            'entities': entities,  # Include collected entities
            'message': 'Confirmation timed out after 2 attempts.',  # Timeout message
        }  # End of timeout return

    except Exception as dialogue_err:  # Catch any unexpected error in the dialogue flow
        print(f'[SIO DM] Error in dialogue manager: {dialogue_err}')  # Log the error message
        return {  # Return error result dictionary
            'status': 'error',  # Mark status as error
            'intent': nlp_result.get('intent', 'Unknown'),  # Include intent if available
            'entities': nlp_result.get('entities', {}),  # Include entities if available
            'message': f'Dialogue error: {dialogue_err}',  # Include error description
        }  # End of error return

# ─── Standalone test runner ───────────────────────────────────────────────────── #
# Test the dialogue manager directly without running the full pipeline           # Description comment
def run_test():  # Define function to test the dialogue manager in isolation
    print('=' * 70)  # Print top banner border
    print('  SIO — Module 3: Dialogue Manager Test  ')  # Print test title banner
    print('=' * 70)  # Print banner separator

    from STT.whisper_stt import load_whisper_model  # Import Whisper model loader for voice input
    print('[SIO DM Test] Loading Whisper model...')  # Log model loading
    whisper_model = load_whisper_model()  # Load the Whisper model for speech recognition
    if not whisper_model:  # Check if Whisper model failed to load
        print('[SIO DM Test] Error: Could not load Whisper model. Exiting test.')  # Log error
        return  # Exit test if model is unavailable

    # Simulate a Fund Transfer NLP result with only recipient filled             # Test description
    simulated_nlp_result = {  # Build a mock NLP result as Module 2 would return
        'intent': 'Fund Transfer',  # Simulate Fund Transfer intent
        'confidence': 0.85,  # Simulate medium-high confidence
        'entities': {  # Simulate partial entities — recipient present but amount missing
            'amount': None,  # Amount is missing — dialogue manager should ask for it
            'recipient': 'Raju',  # Recipient is already known from speech
        },  # End of entities
        'engine': 'Test Simulation',  # Mark as test simulation
    }  # End of simulated NLP result

    print(f'\n[SIO DM Test] Simulated NLP result: {simulated_nlp_result}')  # Print test input
    result = run_dialogue(simulated_nlp_result, whisper_model)  # Run dialogue manager with mock input
    print('\n[SIO DM Test] Final Dialogue Result:')  # Print result header
    print(f'  Status  : {result["status"]}')  # Print final status (confirmed / cancelled / error)
    print(f'  Intent  : {result["intent"]}')  # Print final intent
    print(f'  Entities: {result["entities"]}')  # Print all collected entities
    print(f'  Message : {result["message"]}')  # Print result message
    print('=' * 70)  # Print footer border

if __name__ == '__main__':  # Check if script is run directly from command line
    run_test()  # Run the standalone dialogue manager test
