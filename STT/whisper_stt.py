# SIO Voice-First Banking Assistant - Module 1: Speech to Text (STT)              # File title comment
# This script records audio and transcribes Hindi, English, and Hinglish speech.  # Module description comment
# It automatically detects the language and converts spoken words into text.     # Capability explanation comment
import os  # Import the os module to handle operating system paths and file deletion
import re  # Import the re module for regular expressions used in post-processing transcription
import sys  # Import the sys module to interact with Python interpreter and environment
import time  # Import the time module to provide countdown timers before recording
import tempfile  # Import tempfile module to generate secure temporary file paths
import subprocess  # Import subprocess module to execute pip commands for installing packages
# List of required external Python libraries for recording and transcription     # Description comment
REQUIRED_PACKAGES = [  # Define list of package names needed for the STT module
    "openai-whisper",  # OpenAI Whisper library for speech-to-text transcription
    "sounddevice",  # Sounddevice library to record live audio from microphone
    "scipy",  # Scipy library to write recorded audio arrays into WAV files
    "numpy",  # Numpy library for array manipulation and audio sample conversions
    "imageio-ffmpeg",  # Imageio-ffmpeg library providing bundled ffmpeg binaries
]  # End of the required packages list
# Function to automatically install any missing libraries using pip               # Description comment
def install_required_packages():  # Define function to check and install missing packages
    print("[SIO STT] Checking and installing required dependencies...")  # Print status message to the console
    for package in REQUIRED_PACKAGES:  # Loop through each required package in the list
        try:  # Start try block to check if the package is importable
            import_name = package.replace("-", "_")  # Convert hyphenated names to valid Python module identifiers
            if package == "openai-whisper":  # Check if checking openai-whisper package
                import_name = "whisper"  # Set import name to whisper for openai-whisper
            __import__(import_name)  # Dynamically import the module to verify its availability
            print(f"[SIO STT] Package '{package}' is already installed.")  # Confirm that the package is installed
        except ImportError:  # Handle case where the package is not yet installed
            print(f"[SIO STT] Package '{package}' not found. Installing via pip...")  # Inform user of installation
            try:  # Start try block to execute pip install command
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])  # Run pip install command
                print(f"[SIO STT] Package '{package}' installed successfully.")  # Confirm successful package installation
            except Exception as install_error:  # Handle errors that occur during pip install
                print(f"[SIO STT] Error installing '{package}': {install_error}")  # Print installation error message
# Execute automatic package installation before importing third-party libraries # Description comment
install_required_packages()  # Call the package installer function before loading third-party imports

import numpy as np  # Import numpy for numerical array processing and audio conversion
import sounddevice as sd  # Import sounddevice to record audio input from the user microphone
from scipy.io import wavfile  # Import wavfile from scipy to save and read WAV audio files
import whisper  # Import openai-whisper library to load models and perform transcription
import torch  # Import torch to check for GPU availability and tensor operations
# Helper function to ensure ffmpeg binary is accessible in system PATH           # Description comment
def configure_ffmpeg_path():  # Define function to configure ffmpeg binary path for Whisper
    try:  # Start try block to locate and configure ffmpeg
        import imageio_ffmpeg  # Import imageio_ffmpeg to find the location of bundled ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()  # Get full path of the bundled ffmpeg executable
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)  # Get directory containing the ffmpeg binary
        if ffmpeg_dir not in os.environ.get("PATH", ""):  # Check if ffmpeg folder is not in PATH
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")  # Add folder to PATH
        standard_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")  # Define standard ffmpeg executable path
        if not os.path.exists(standard_ffmpeg) and os.path.exists(ffmpeg_exe):  # If standard name missing
            import shutil  # Import shutil to copy the binary
            shutil.copyfile(ffmpeg_exe, standard_ffmpeg)  # Copy versioned binary to standard ffmpeg.exe
    except Exception as ffmpeg_error:  # Catch any exception during ffmpeg path configuration
        print(f"[SIO STT] Note: ffmpeg configuration notice: {ffmpeg_error}")  # Print notice if configuration fails

configure_ffmpeg_path()  # Run ffmpeg configuration helper before using Whisper
# Audio recording and model configuration constants                              # Description comment
MODEL_NAME = "base"  # Whisper model size ("base" for fast speed, "small" for maximum accuracy)
SAMPLE_RATE = 16000  # Set audio sample rate to 16000 Hz as expected by OpenAI Whisper
DURATION = 5  # Set recording duration to 5 seconds as requested
# Function to verify whether an audio input device (microphone) is available    # Description comment
def check_microphone_availability():  # Define function to check for microphone hardware
    try:  # Start try block to query audio devices
        devices = sd.query_devices()  # Retrieve list of all audio devices on the system
        input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]  # Filter input devices
        if not input_devices:  # Check if list of input devices is empty
            print("[SIO STT] Error: No microphone input device found on this system.")  # Print error message
            print("[SIO STT] Please connect a microphone or headset and try again.")  # Print guidance
            return None  # Return None indicating no microphone was found
        default_input = sd.query_devices(kind='input')  # Query the default recording device
        print(f"[SIO STT] Microphone detected: {default_input.get('name', 'Default Microphone')}")  # Print name
        return default_input  # Return default input device details
    except Exception as mic_error:  # Catch any audio hardware query errors
        print(f"[SIO STT] Error detecting microphone: {mic_error}")  # Print detailed error message
        print("[SIO STT] Please check your audio drivers and microphone permissions.")  # Print guidance
        return None  # Return None indicating microphone check failed
# Function to record 5 seconds of audio and save it to a temporary WAV file      # Description comment
def record_audio_to_wav():  # Define function to record audio and save as temporary WAV
    mic_info = check_microphone_availability()  # Verify microphone availability before recording
    if mic_info is None:  # Check if microphone is missing or unavailable
        return None  # Return None if microphone is not available
    temp_wav_path = None  # Initialize variable to store temporary WAV file path
    try:  # Start try block for audio recording and file creation
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:  # Create temporary file
            temp_wav_path = temp_file.name  # Store absolute path of the temporary file
        channels_to_record = min(2, mic_info.get('max_input_channels', 1))  # Capture stereo if array to prevent blank channels
        print("[SIO STT] Get ready to speak in:")  # Give user a visual countdown prompt
        for count in range(3, 0, -1):  # Loop through 3 second countdown numbers
            print(f"[SIO STT]   {count}...")  # Display countdown second
            time.sleep(1)  # Pause for 1 second per countdown count
        print(f"[SIO STT] >>> SPEAK NOW! (Hindi / English / Hinglish for {DURATION}s) <<<")  # Prompt user to speak
        raw_audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=channels_to_record, dtype='float32')  # Record
        sd.wait()  # Wait for the full 5 seconds of recording to complete
        print("[SIO STT] Recording finished! Processing audio...")  # Notify user recording has completed
        if channels_to_record > 1:  # Check if multi-channel audio was recorded
            peak_ch0 = float(np.max(np.abs(raw_audio[:, 0])))  # Measure peak audio amplitude on first channel
            peak_ch1 = float(np.max(np.abs(raw_audio[:, 1])))  # Measure peak audio amplitude on second channel
            if peak_ch1 > (peak_ch0 * 2.0):  # Check if channel 1 has significantly stronger signal than channel 0
                mono_audio = raw_audio[:, 1]  # Select channel 1 as primary audio channel
            elif peak_ch0 > (peak_ch1 * 2.0):  # Check if channel 0 has significantly stronger signal than channel 1
                mono_audio = raw_audio[:, 0]  # Select channel 0 as primary audio channel
            else:  # If both channels have balanced signal
                mono_audio = np.mean(raw_audio, axis=1)  # Average both channels into a single mono audio channel
        else:  # If recorded with single channel
            mono_audio = raw_audio.flatten()  # Flatten single channel array into 1D audio array
        peak_volume = float(np.max(np.abs(mono_audio)))  # Calculate peak audio amplitude across the entire recording
        if peak_volume < 0.005:  # Check if volume is extremely low suggesting muted or distant microphone
            print("[SIO STT] Notice: Audio signal was very faint. Speaking louder or closer to mic is recommended.")  # Warn user
        if peak_volume > 0.001:  # Check if there is enough sound signal to perform normalization
            mono_audio = (mono_audio / peak_volume) * 0.95  # Normalize audio amplitude to 95 percent of full dynamic range
        int16_audio = (mono_audio * 32767.0).astype(np.int16)  # Convert normalized float32 samples to 16-bit PCM format
        wavfile.write(temp_wav_path, SAMPLE_RATE, int16_audio)  # Save 16-bit PCM audio samples to temporary WAV file
        print(f"[SIO STT] Audio saved temporarily at: {temp_wav_path}")  # Log temporary file path
        return temp_wav_path  # Return path of saved temporary WAV file
    except Exception as record_error:  # Catch any error occurring during recording or saving
        print(f"[SIO STT] Error during audio recording: {record_error}")  # Print recording error message
        if temp_wav_path and os.path.exists(temp_wav_path):  # Check if temporary file exists on error
            try:  # Start try block for cleaning up incomplete file
                os.remove(temp_wav_path)  # Delete incomplete temporary WAV file
            except OSError:  # Catch any OS error during cleanup
                pass  # Ignore file cleanup error
        return None  # Return None indicating recording failed
# Function to load OpenAI Whisper base model with graceful error handling        # Description comment
def load_whisper_model():  # Define function to load the Whisper model
    try:  # Start try block to load OpenAI Whisper base model
        print(f"[SIO STT] Loading OpenAI Whisper '{MODEL_NAME}' model...")  # Notify user that model loading started
        model = whisper.load_model(MODEL_NAME)  # Load the OpenAI Whisper model weights into memory
        print(f"[SIO STT] OpenAI Whisper '{MODEL_NAME}' model loaded successfully.")  # Confirm successful model loading
        return model  # Return the loaded Whisper model instance
    except Exception as load_error:  # Catch any error when loading the model
        print(f"[SIO STT] Error: Failed to load OpenAI Whisper model: {load_error}")  # Print loading error
        print("[SIO STT] Please check your internet connection or PyTorch installation.")  # Print guidance
        return None  # Return None indicating model loading failed
# Post-processing function to clean common Whisper hallucination artifacts       # Description comment
# Whisper sometimes adds leading filler words, wrong currency formatting, or trailing noise # Description comment
def post_process_transcription(text):  # Define function to clean and normalize raw Whisper output
    if not text:  # Check if the text is empty or None before processing
        return text  # Return unchanged if text is empty or None
    cleaned = text.strip()  # Remove any leading or trailing whitespace from the raw transcript
    # Step 1: Remove common leading filler words Whisper inserts at the start    # Step description comment
    # e.g. "And Raju..." should become "Raju..." when user said "Send Raju..."   # Example comment
    cleaned = re.sub(  # Apply regex substitution to strip leading filler words
        r'^\s*(?:and|oh|um|uh|so|well|now|okay|ok|hmm|hey|alright|right)\s+',  # Pattern: leading filler words
        '',  # Replace with empty string to remove the filler word completely
        cleaned,  # Apply on the current cleaned text
        flags=re.IGNORECASE  # Ignore case so AND, And, and all match
    )  # End of leading filler word removal
    # Step 2: Normalize Indian currency formatting that Whisper may add          # Step description comment
    # e.g. "Rs.5,00,000" should become "500000" and "Rs.500,000" → "500000"     # Example comment
    # First pass: remove Indian comma grouping inside Rs.X,XX,XXX style amounts  # Comment
    cleaned = re.sub(  # Apply regex to normalize Rs. currency amounts
        r'\bRs\.?\s*([\d,]+)',  # Pattern: Rs or Rs. followed by digits with commas
        lambda m: m.group(1).replace(',', ''),  # Lambda: remove all commas from the number portion
        cleaned,  # Apply on current cleaned text
        flags=re.IGNORECASE  # Match Rs, RS, rs in any case
    )  # End of Rs. normalization
    # Second pass: plain numbers with comma grouping e.g. 5,000 → 5000          # Comment
    cleaned = re.sub(  # Apply regex to remove commas from standalone formatted numbers
        r'\b(\d{1,3}(?:,\d{2,3})+)\b',  # Pattern: numbers formatted with Indian comma grouping
        lambda m: m.group(0).replace(',', ''),  # Lambda: strip all commas from the matched number
        cleaned  # Apply on current cleaned text
    )  # End of plain number comma removal
    # Step 3: Remove trailing noise artifacts that Whisper sometimes appends     # Step description comment
    # e.g. "Send Raju 5000 rupees to Rs." → "Send Raju 5000 rupees"             # Example comment
    cleaned = re.sub(  # Apply regex to strip trailing Rs./INR currency artifacts
        r'\s+(?:to\s+)?(?:Rs\.?|INR)\s*$',  # Pattern: trailing "to Rs." or "Rs." at end of string
        '',  # Replace with empty string to remove trailing noise
        cleaned,  # Apply on current cleaned text
        flags=re.IGNORECASE  # Match case-insensitively
    )  # End of trailing noise removal
    cleaned = cleaned.strip()  # Final strip of any remaining whitespace after all substitutions
    return cleaned  # Return the fully cleaned and normalized transcription text
# Function to transcribe audio in Hindi, English, or Hinglish using Whisper       # Description comment
# Includes language validation: falls back to Hindi if Whisper detects wrong language # Description comment
SUPPORTED_LANGUAGES = {"hi", "en"}  # Set of supported language codes for SIO (Hindi and English)
def transcribe_audio(model, wav_path):  # Define function to transcribe audio in Hindi, English, or Hinglish
    try:  # Start try block for transcribing audio
        print("[SIO STT] Transcribing audio (auto-detecting Hindi / English / Hinglish)...")  # Notify user
        use_fp16 = torch.cuda.is_available()  # Determine whether CUDA GPU is available for FP16 inference
        # Conditioning prompt providing mixed Hindi, English, and Hinglish banking context to guide Whisper
        # Examples of fund-transfer style commands are included to help Whisper recognize them correctly
        multilingual_prompt = (  # Multi-line prompt string for guiding Whisper context
            "Hello, नमस्ते, mera naam Shubham hai, account balance check, paise transfer, "  # Greeting and balance context
            "ATM PIN, pension. Send Raju 5000 rupees. Raju ko 5000 bhejo. "  # Fund transfer examples
            "Mera balance kitna hai? Paisa bhejo."  # Balance and payment examples
        )  # End of multilingual context prompt
        try:  # Start nested try block to attempt transcription using file path directly
            # First attempt: auto-detect language (works well when user speaks clearly)
            result = model.transcribe(wav_path, task="transcribe", initial_prompt=multilingual_prompt, temperature=0.0, beam_size=5, fp16=use_fp16)  # Transcribe
        except Exception:  # If transcription via file path fails (e.g. ffmpeg not in PATH)
            _, audio_data = wavfile.read(wav_path)  # Read audio data array directly from WAV file
            audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize int16 audio array to float32
            if audio_float.ndim > 1:  # Check if audio has multiple channels
                audio_float = audio_float.mean(axis=1)  # Convert multi-channel audio to mono
            result = model.transcribe(audio_float, task="transcribe", initial_prompt=multilingual_prompt, temperature=0.0, beam_size=5, fp16=use_fp16)  # Transcribe
        detected_language = result.get("language", "unknown")  # Retrieve the detected language code from Whisper
        transcribed_text = result.get("text", "").strip()  # Extract and strip transcribed text from result
        # Language validation: if Whisper detected a non-supported language (e.g. Telugu, Tamil, Marathi)
        # it usually means background noise confused it — re-run forcing Hindi for better accuracy
        if detected_language not in SUPPORTED_LANGUAGES:  # Check if detected language is not Hindi or English
            print(f"[SIO STT] Detected unexpected language '{detected_language}'. Re-transcribing in Hindi...")  # Warn user
            hindi_banking_prompt = "नमस्ते, मेरा खाता, बैलेंस, पैसे ट्रांसफर, एटीएम, रिचार्ज, बिल।"  # Hindi-only context prompt
            try:  # Start try block for Hindi forced retranscription via file path
                result = model.transcribe(wav_path, language="hi", task="transcribe", initial_prompt=hindi_banking_prompt, temperature=0.2, beam_size=5, fp16=use_fp16)  # Force Hindi
            except Exception:  # If file path transcription fails again
                _, audio_data = wavfile.read(wav_path)  # Read audio data directly from WAV file
                audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize audio to float32 range
                if audio_float.ndim > 1:  # Check if audio is multi-channel
                    audio_float = audio_float.mean(axis=1)  # Mix down to mono
                result = model.transcribe(audio_float, language="hi", task="transcribe", initial_prompt=hindi_banking_prompt, temperature=0.2, beam_size=5, fp16=use_fp16)  # Force Hindi
            detected_language = result.get("language", "hi")  # Update detected language after forced Hindi pass
            transcribed_text = result.get("text", "").strip()  # Update transcribed text from Hindi pass
        # Apply post-processing to remove Whisper hallucinations before returning  # Description comment
        transcribed_text = post_process_transcription(transcribed_text)  # Clean raw Whisper output
        return transcribed_text, detected_language  # Return final cleaned transcribed text and language code
    except Exception as transcribe_error:  # Catch any unexpected error during transcription
        print(f"[SIO STT] Error during transcription: {transcribe_error}")  # Print transcription error message
        return None, None  # Return None indicating transcription failed
# Function to delete the temporary WAV file after transcription                 # Description comment
def remove_temporary_file(wav_path):  # Define function to delete temporary WAV file
    if wav_path and os.path.exists(wav_path):  # Check if temporary file path exists on disk
        try:  # Start try block to safely delete the temporary file
            os.remove(wav_path)  # Remove the temporary WAV file from disk
            print(f"[SIO STT] Deleted temporary file: {wav_path}")  # Confirm file deletion to user
        except Exception as delete_error:  # Catch any error during file deletion
            print(f"[SIO STT] Warning: Could not delete temporary file: {delete_error}")  # Print warning
# Main execution function orchestrating the full Speech to Text workflow         # Description comment
def main():  # Define main function for executing Module 1 STT
    print("=" * 65)  # Print decorative top banner border
    print("  SIO Voice Banking Assistant - Module 1: Speech to Text (Multilingual)  ")  # Print module title banner
    print("=" * 65)  # Print decorative banner divider
    model = load_whisper_model()  # Load the OpenAI Whisper base model
    if model is None:  # Check if Whisper model failed to load
        print("[SIO STT] Exiting: Whisper model could not be initialized.")  # Print error exit message
        return  # Stop execution because model is unavailable
    temp_wav = record_audio_to_wav()  # Record 5 seconds of audio and save to temporary WAV file
    if temp_wav is None:  # Check if audio recording failed or microphone was missing
        print("[SIO STT] Exiting: Audio recording could not be completed.")  # Print error exit message
        return  # Stop execution because recording failed
    try:  # Start try block for transcription and display
        transcribed_text, detected_lang = transcribe_audio(model, temp_wav)  # Transcribe audio to text
        if transcribed_text:  # Check if transcribed text is not empty
            lang_label = "Hindi" if detected_lang == "hi" else ("English" if detected_lang == "en" else detected_lang)  # Friendly label
            print("-" * 65)  # Print section divider line
            print(f"Detected Language : {lang_label} ({detected_lang})")  # Print detected language to console
            print(f"Transcribed Text  : {transcribed_text}")  # Print transcribed text to console
            print("-" * 65)  # Print section divider line
        else:  # If transcribed text is empty
            print("[SIO STT] No speech detected or transcription returned empty text.")  # Print notice
    finally:  # Ensure temporary file is always cleaned up after transcription
        remove_temporary_file(temp_wav)  # Delete the temporary WAV file from disk

if __name__ == "__main__":  # Check whether script is run directly from command line
    main()  # Run the main STT workflow function
