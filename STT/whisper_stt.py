# SIO Voice-First Banking Assistant - Module 1: Speech to Text (STT)              # File title comment
# This script records audio and transcribes Hindi, English, and Hinglish speech.  # Module description comment
# Key improvements: one-time mic selection, noise reduction, voice activity detect # Improvement summary comment

import os  # Import os module for file and path operations
import re  # Import re module for post-processing text with regular expressions
import sys  # Import sys module for runtime configuration
import time  # Import time module for countdown timers and pauses
import tempfile  # Import tempfile module for secure temporary file creation
import subprocess  # Import subprocess module for pip installs

# Reconfigure stdout and stderr to UTF-8 so Hindi prints correctly on Windows    # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if stdout supports reconfiguration
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout to UTF-8
if hasattr(sys.stderr, 'reconfigure'):  # Check if stderr supports reconfiguration
    sys.stderr.reconfigure(encoding='utf-8')  # Set stderr to UTF-8

# ── Auto-install required packages ───────────────────────────────────────────── #
REQUIRED_PACKAGES = [  # List all packages the STT module needs
    ('openai-whisper', 'whisper'),  # Whisper for speech transcription
    ('sounddevice',    'sounddevice'),  # sounddevice for microphone recording
    ('scipy',          'scipy'),  # scipy for WAV file I/O
    ('numpy',          'numpy'),  # numpy for audio array math
    ('imageio-ffmpeg', 'imageio_ffmpeg'),  # ffmpeg binary bundled for Whisper
    ('noisereduce',    'noisereduce'),  # noisereduce for background noise removal
]  # End of required packages list

def _ensure_packages():  # Define function to auto-install any missing packages
    for pip_name, import_name in REQUIRED_PACKAGES:  # Loop through each package
        try:  # Try importing the package to check if it exists
            __import__(import_name)  # Attempt import
        except ImportError:  # Package is missing — install it
            print(f'[SIO STT] Installing {pip_name}...')  # Inform user
            try:  # Try pip install
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name, '--quiet'])  # Install
            except Exception as e:  # Catch install errors
                print(f'[SIO STT] Could not install {pip_name}: {e}')  # Print warning

_ensure_packages()  # Run package check before importing anything else

# ── Third-party imports (all available after _ensure_packages) ───────────────── #
import numpy as np  # Import numpy for array math and audio processing
import sounddevice as sd  # Import sounddevice for microphone recording
from scipy.io import wavfile  # Import wavfile for reading and writing WAV files
import whisper  # Import OpenAI Whisper for speech-to-text transcription
import torch  # Import torch for GPU detection

try:  # Try importing noisereduce — gracefully skip if unavailable
    import noisereduce as nr  # Import noisereduce for spectral noise subtraction
    NR_AVAILABLE = True  # Mark noisereduce as available
except ImportError:  # Catch if noisereduce is not installed
    NR_AVAILABLE = False  # Mark as unavailable — will skip noise reduction

# ── ffmpeg path configuration ─────────────────────────────────────────────────── #
def _configure_ffmpeg():  # Define helper to add bundled ffmpeg to system PATH
    try:  # Try locating the bundled ffmpeg binary
        import imageio_ffmpeg  # Import imageio_ffmpeg to find bundled binary
        exe = imageio_ffmpeg.get_ffmpeg_exe()  # Get full path to ffmpeg executable
        d = os.path.dirname(exe)  # Get directory containing ffmpeg
        if d not in os.environ.get('PATH', ''):  # Check if not already in PATH
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')  # Add to PATH
        std = os.path.join(d, 'ffmpeg.exe')  # Build standard ffmpeg.exe path
        if not os.path.exists(std) and os.path.exists(exe):  # Check if standard name missing
            import shutil  # Import shutil for file copying
            shutil.copyfile(exe, std)  # Copy versioned binary to standard name
    except Exception as e:  # Catch any ffmpeg setup error
        print(f'[SIO STT] ffmpeg notice: {e}')  # Print notice (non-fatal)

_configure_ffmpeg()  # Run ffmpeg setup at import time

# ── Constants ─────────────────────────────────────────────────────────────────── #
MODEL_NAME   = 'base'   # Whisper model — 'base' is fast; upgrade to 'small' for more accuracy
SAMPLE_RATE  = 16000    # 16 kHz — Whisper's native sample rate for best accuracy
DURATION     = 7        # Maximum recording duration in seconds (longer = catches full sentences)
SILENCE_THRESHOLD = 0.008  # RMS energy below this level is treated as silence
SILENCE_AFTER_SPEECH_SEC = 1.5  # Stop early after this many seconds of silence post-speech
SUPPORTED_LANGUAGES = {'hi', 'en'}  # Language codes accepted by SIO

# ── ONE-TIME device selection (cached at startup) ─────────────────────────────── #
# Device is selected ONCE when the module loads — never re-scanned per recording  # Description comment
_SELECTED_DEVICE_ID   = None  # Cached microphone device ID (int or None)
_SELECTED_DEVICE_NAME = None  # Cached microphone device name for display

def _select_microphone():  # Define function to pick the best mic once at startup
    """
    Picks the best available input device using a two-stage approach:
      Stage 1 — Name priority: Bluetooth and USB headsets ranked above built-in mics.
      Stage 2 — Signal test: Among equally-ranked devices, pick the one with highest peak.
    Result is cached in module-level variables so this runs ONLY ONCE per session.
    """
    global _SELECTED_DEVICE_ID, _SELECTED_DEVICE_NAME  # Access module-level cache variables

    # ── Priority keywords — higher number = higher priority ─────────────────────── #
    PRIORITY_KEYWORDS = [  # List of (keyword, priority_score) tuples for device name matching
        ('bluetooth',   5),  # Bluetooth headsets are best for voice capture
        ('headset',     5),  # Wired headsets are also great
        ('buds',        5),  # In-ear buds (e.g. realme Buds) have close-mic advantage
        ('usb',         4),  # USB microphones are typically high quality
        ('external',    4),  # External mics rank above built-in
        ('webcam',      3),  # Webcam mics are acceptable
        ('array',       2),  # Mic arrays (like Realtek Array) are often lowest quality
        ('microphone',  1),  # Generic built-in microphone fallback
    ]  # End of priority keyword list

    try:  # Start try block for device enumeration
        all_devices = sd.query_devices()  # Get list of all audio devices
        candidates = []  # List to collect (device_id, name, priority, peak) tuples

        print('[SIO STT] Selecting best microphone (one-time setup)...')  # Inform user

        for idx, dev in enumerate(all_devices):  # Loop through each audio device
            if dev.get('max_input_channels', 0) < 1:  # Skip output-only devices
                continue  # Move to next device

            name_lower = dev['name'].lower()  # Lowercase device name for keyword matching

            # Calculate priority score based on device name keywords              # Comment
            priority = 0  # Start with zero priority
            for keyword, score in PRIORITY_KEYWORDS:  # Check each keyword
                if keyword in name_lower:  # If keyword found in device name
                    priority = max(priority, score)  # Use highest matching score

            # Quick 0.3-second signal test to measure actual audio level          # Comment
            # This runs once at startup so it's acceptable                        # Comment
            try:  # Try recording from this device
                test = sd.rec(  # Record a short burst to measure signal level
                    int(0.3 * SAMPLE_RATE),  # 0.3 seconds of samples
                    samplerate=SAMPLE_RATE,  # 16 kHz sample rate
                    channels=1,  # Mono channel for simplicity
                    dtype='float32',  # Float format for peak measurement
                    device=idx  # Device being tested
                )  # End of rec call
                sd.wait()  # Wait for test recording to finish
                peak = float(np.max(np.abs(test)))  # Measure peak amplitude
                candidates.append((idx, dev['name'], priority, peak))  # Add to list
            except Exception:  # Skip devices that fail to open
                continue  # Move to next device

        if not candidates:  # Check if no devices were found
            print('[SIO STT] ERROR: No usable microphone found!')  # Print error
            return  # Exit without setting cache

        # Sort by priority (descending), then by peak amplitude (descending)     # Comment
        candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)  # Sort best first

        best_id, best_name, best_pri, best_peak = candidates[0]  # Pick the top candidate
        _SELECTED_DEVICE_ID   = best_id    # Cache the chosen device ID
        _SELECTED_DEVICE_NAME = best_name  # Cache the chosen device name

        print(f'[SIO STT] ✅ Microphone selected: [{best_id}] {best_name}')  # Confirm selection
        print(f'[SIO STT]    Priority={best_pri}, Peak={best_peak:.5f}')  # Print details

        if best_peak < 0.001:  # Warn if even the best device has almost no signal
            print('[SIO STT] ⚠️  Warning: Selected mic has very low signal.')  # Warn user
            print('[SIO STT]    Check Windows: Settings > Privacy > Microphone — allow app access.')  # Guidance

    except Exception as e:  # Catch any unexpected error during device selection
        print(f'[SIO STT] Error selecting microphone: {e}')  # Log error

# Run device selection ONCE when this module is first imported                   # Description comment
_select_microphone()  # Select and cache the best microphone device at startup

# ── Noise reduction ───────────────────────────────────────────────────────────── #
def _reduce_noise(audio_float32):  # Define function to remove background noise from audio
    """
    Applies spectral noise reduction using noisereduce.
    Uses the first 0.3 seconds of the recording as the noise profile
    (captures background hum, AC noise, keyboard sounds etc.)
    """
    if not NR_AVAILABLE:  # Check if noisereduce library is installed
        return audio_float32  # Return unchanged audio if library not available
    try:  # Start try block for noise reduction
        noise_sample_len = int(0.3 * SAMPLE_RATE)  # 0.3 seconds = noise profile length
        noise_sample = audio_float32[:noise_sample_len]  # Slice first 0.3 seconds as noise reference
        reduced = nr.reduce_noise(  # Apply spectral subtraction noise reduction
            y=audio_float32,  # Full audio signal to clean
            y_noise=noise_sample,  # Noise reference sample from start of recording
            sr=SAMPLE_RATE,  # Sample rate needed for frequency analysis
            prop_decrease=0.85,  # Reduce noise by 85% — aggressive but keeps voice clear
            stationary=False,  # Non-stationary mode handles changing background noise better
        )  # End of reduce_noise call
        return reduced.astype(np.float32)  # Return cleaned audio as float32
    except Exception as nr_err:  # Catch any error in noise reduction
        print(f'[SIO STT] Noise reduction skipped: {nr_err}')  # Log and continue
        return audio_float32  # Return original audio if noise reduction fails

# ── Voice Activity Detection (VAD) recording ──────────────────────────────────── #
def _record_with_vad():  # Define function to record audio with voice activity detection
    """
    Records audio in small 100ms chunks.
    - Waits up to DURATION seconds for voice to start.
    - Once voice is detected, continues recording.
    - Stops automatically after SILENCE_AFTER_SPEECH_SEC of silence post-speech.
    - Maximum total recording is DURATION seconds.
    This means the user does not need to wait the full 5-7 seconds every time.
    """
    if _SELECTED_DEVICE_ID is None:  # Check if a device was successfully selected
        print('[SIO STT] No microphone available for recording.')  # Error message
        return None  # Return None if no device is selected

    chunk_size = int(0.1 * SAMPLE_RATE)  # Each chunk is 100 milliseconds of audio
    max_chunks = int(DURATION / 0.1)  # Maximum number of chunks for the full duration
    silence_chunks_needed = int(SILENCE_AFTER_SPEECH_SEC / 0.1)  # Chunks of silence to trigger stop

    all_chunks = []  # List to accumulate all recorded audio chunks
    speech_started = False  # Flag: has the user started speaking yet
    silence_chunk_count = 0  # Counter for consecutive silent chunks after speech

    print(f'[SIO STT] 🎙️  Listening... speak now! (up to {DURATION}s, stops when you finish)')  # Prompt

    try:  # Start try block for VAD recording loop
        for _ in range(max_chunks):  # Loop for maximum allowed number of chunks
            chunk = sd.rec(  # Record one 100ms chunk from the selected device
                chunk_size,  # Number of samples in this chunk
                samplerate=SAMPLE_RATE,  # 16 kHz sample rate
                channels=1,  # Mono recording
                dtype='float32',  # Float format for RMS measurement
                device=_SELECTED_DEVICE_ID  # Use the cached best device
            )  # End of rec call
            sd.wait()  # Wait for this 100ms chunk to finish recording
            chunk_flat = chunk.flatten()  # Flatten to 1D array
            all_chunks.append(chunk_flat)  # Add chunk to full recording list
            rms = float(np.sqrt(np.mean(chunk_flat ** 2)))  # Calculate RMS energy of chunk

            if rms > SILENCE_THRESHOLD:  # Check if this chunk contains voice energy
                speech_started = True  # Mark that speech has begun
                silence_chunk_count = 0  # Reset silence counter whenever voice is heard
            elif speech_started:  # If speech started but this chunk is silent
                silence_chunk_count += 1  # Increment silence counter
                if silence_chunk_count >= silence_chunks_needed:  # Check if silence is long enough
                    print('[SIO STT] ✅ Voice detected and finished. Processing...')  # Inform user
                    break  # Stop recording — user has finished speaking

        if not speech_started:  # Check if no voice was ever detected
            print('[SIO STT] ⚠️  No voice detected. Please speak clearly into the microphone.')  # Warn
            return None  # Return None if no voice was captured

        full_audio = np.concatenate(all_chunks)  # Join all chunks into one audio array
        return full_audio  # Return the complete raw audio array

    except Exception as rec_err:  # Catch any recording error
        print(f'[SIO STT] Recording error: {rec_err}')  # Log error
        return None  # Return None on error

# ── Main recording function ───────────────────────────────────────────────────── #
def record_audio_to_wav():  # Define public function to record and save audio as WAV
    """
    Records user's voice using VAD, applies noise reduction,
    normalizes volume, and saves to a temporary WAV file.
    Returns the path to the temporary WAV file, or None on failure.
    """
    print('[SIO STT] Get ready to speak...')  # Inform user we are preparing to listen
    raw_audio = _record_with_vad()  # Record with voice activity detection
    if raw_audio is None:  # Check if recording failed or no voice detected
        return None  # Return None if recording failed

    # ── Apply noise reduction ──────────────────────────────────────────────────── #
    if NR_AVAILABLE:  # Check if noise reduction library is available
        print('[SIO STT] Applying noise reduction...')  # Inform user
    clean_audio = _reduce_noise(raw_audio)  # Remove background noise from recording

    # ── Normalize volume ───────────────────────────────────────────────────────── #
    peak = float(np.max(np.abs(clean_audio)))  # Measure peak amplitude of cleaned audio
    if peak > 0.001:  # Check if there is enough signal to normalize
        clean_audio = (clean_audio / peak) * 0.95  # Normalize to 95% of full dynamic range
    else:  # Audio is still too quiet even after noise reduction
        print('[SIO STT] ⚠️  Audio is very quiet. Try speaking louder or closer.')  # Warn user

    # ── Save to temporary WAV file ─────────────────────────────────────────────── #
    temp_wav_path = None  # Initialize temp file path
    try:  # Start try block for file saving
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:  # Create temp file
            temp_wav_path = tmp.name  # Store path for later use
        int16_audio = (clean_audio * 32767.0).astype(np.int16)  # Convert to 16-bit PCM integer format
        wavfile.write(temp_wav_path, SAMPLE_RATE, int16_audio)  # Write audio to WAV file
        print(f'[SIO STT] Audio saved: {temp_wav_path}')  # Log saved path
        return temp_wav_path  # Return path to the saved WAV file
    except Exception as save_err:  # Catch file saving errors
        print(f'[SIO STT] Error saving audio: {save_err}')  # Log error
        if temp_wav_path and os.path.exists(temp_wav_path):  # Check if partial file exists
            try:  # Try deleting partial file
                os.remove(temp_wav_path)  # Delete the partial file
            except OSError:  # Ignore deletion errors
                pass  # Continue silently
        return None  # Return None on save failure

# ── Whisper model loader ──────────────────────────────────────────────────────── #
def load_whisper_model():  # Define function to load the Whisper speech recognition model
    try:  # Start try block for model loading
        print(f"[SIO STT] Loading OpenAI Whisper '{MODEL_NAME}' model...")  # Inform user
        model = whisper.load_model(MODEL_NAME)  # Load Whisper model into memory
        print(f"[SIO STT] ✅ Whisper '{MODEL_NAME}' model ready.")  # Confirm success
        return model  # Return loaded model instance
    except Exception as e:  # Catch loading errors
        print(f'[SIO STT] Error loading Whisper: {e}')  # Log error
        return None  # Return None on failure

# ── Post-processing: clean Whisper hallucinations ────────────────────────────── #
def post_process_transcription(text):  # Define function to clean raw Whisper output
    if not text:  # Check if text is empty
        return text  # Return unchanged if empty
    cleaned = text.strip()  # Strip leading and trailing whitespace
    # Remove leading filler words Whisper sometimes adds (e.g. "And Raju...")    # Comment
    cleaned = re.sub(  # Apply regex to remove leading filler words
        r'^\s*(?:and|oh|um|uh|so|well|now|okay|ok|hmm|hey|alright|right)\s+',  # Filler pattern
        '', cleaned, flags=re.IGNORECASE  # Case-insensitive match
    )  # End of filler removal
    # Normalize Indian currency formatting "Rs.5,000" → "5000"                  # Comment
    cleaned = re.sub(  # Remove Rs. prefix and commas from currency amounts
        r'\bRs\.?\s*([\d,]+)',  # Match Rs. followed by digits with commas
        lambda m: m.group(1).replace(',', ''),  # Remove all commas from the number
        cleaned, flags=re.IGNORECASE  # Case-insensitive
    )  # End of currency normalization
    # Remove plain comma grouping from numbers e.g. "5,000" → "5000"            # Comment
    cleaned = re.sub(  # Remove comma grouping from standalone numbers
        r'\b(\d{1,3}(?:,\d{2,3})+)\b',  # Match comma-grouped number pattern
        lambda m: m.group(0).replace(',', ''),  # Strip all commas from match
        cleaned  # Apply on cleaned text
    )  # End of plain number cleanup
    # Strip trailing noise e.g. "...5000 rupees to Rs."                          # Comment
    cleaned = re.sub(  # Remove trailing currency noise artifacts
        r'\s+(?:to\s+)?(?:Rs\.?|INR)\s*$',  # Match trailing Rs./INR patterns
        '', cleaned, flags=re.IGNORECASE  # Case-insensitive
    )  # End of trailing noise removal
    return cleaned.strip()  # Return final cleaned text

# ── Transcription function ────────────────────────────────────────────────────── #
def transcribe_audio(model, wav_path):  # Define function to transcribe WAV file to text
    try:  # Start try block for transcription
        print('[SIO STT] Transcribing speech...')  # Inform user
        use_fp16 = torch.cuda.is_available()  # Use FP16 only if CUDA GPU is available
        # Rich multilingual context prompt for better Whisper accuracy             # Comment
        multilingual_prompt = (  # Build multi-line context prompt string
            'Hello, namaste, mera naam Shubham hai, account balance check, '  # Greeting context
            'paise transfer karo, ATM PIN, pension, recharge, bill. '  # Banking command context
            'Send Raju 5000 rupees. Raju ko 5000 bhejo. '  # Fund transfer examples
            'Bijli ka bill bharo. Mera balance kitna hai?'  # Bill and balance examples
        )  # End of prompt string

        try:  # Try transcribing using file path (preferred — uses ffmpeg pipeline)
            result = model.transcribe(  # Call Whisper transcribe on the WAV file
                wav_path,  # Path to the WAV file
                task='transcribe',  # Transcribe mode (not translate)
                initial_prompt=multilingual_prompt,  # Context prompt to guide Whisper
                temperature=0.0,  # Zero temperature = deterministic, most confident output
                beam_size=5,  # Beam size 5 for better accuracy vs speed trade-off
                best_of=5,  # Pick best result from 5 candidates
                condition_on_previous_text=False,  # Avoid hallucination loops
                fp16=use_fp16  # Use FP16 on GPU for speed
            )  # End of transcribe call
        except Exception:  # If file path fails (e.g. ffmpeg not found) fallback to array
            _, audio_data = wavfile.read(wav_path)  # Read raw audio data from WAV file
            audio_float = audio_data.astype(np.float32) / 32768.0  # Convert int16 to float32
            if audio_float.ndim > 1:  # Check if audio is multi-channel
                audio_float = audio_float.mean(axis=1)  # Mix down to mono
            result = model.transcribe(  # Transcribe using raw audio array instead
                audio_float,  # Float32 audio array
                task='transcribe',  # Transcribe mode
                initial_prompt=multilingual_prompt,  # Same context prompt
                temperature=0.0,  # Deterministic output
                beam_size=5,  # Beam search size
                best_of=5,  # Pick best of 5
                condition_on_previous_text=False,  # Disable previous text conditioning
                fp16=use_fp16  # FP16 if GPU available
            )  # End of array transcribe call

        detected_lang = result.get('language', 'unknown')  # Get detected language code
        transcribed_text = result.get('text', '').strip()  # Get and strip transcribed text

        # Check Whisper's confidence — if no_speech_prob is high, speech was not heard # Comment
        no_speech_prob = result.get('segments', [{}])[0].get('no_speech_prob', 0.0) if result.get('segments') else 0.0  # Get silence prob
        if no_speech_prob > 0.7:  # If more than 70% chance this is silence/noise
            print(f'[SIO STT] ⚠️  Low confidence transcription (no_speech_prob={no_speech_prob:.2f}). Try again.')  # Warn
            return None, None  # Return None to trigger retry at caller level

        # Language validation: if an unsupported language detected, retry in Hindi # Comment
        if detected_lang not in SUPPORTED_LANGUAGES:  # Check if language is Hindi or English
            print(f"[SIO STT] Detected '{detected_lang}' — re-transcribing in Hindi...")  # Log retry
            hindi_prompt = 'नमस्ते, मेरा खाता, बैलेंस, पैसे ट्रांसफर, एटीएम, रिचार्ज, बिल।'  # Hindi prompt
            try:  # Try Hindi-forced transcription via file path
                result = model.transcribe(wav_path, language='hi', task='transcribe',  # Force Hindi
                                          initial_prompt=hindi_prompt, temperature=0.2,  # Slightly higher temp
                                          beam_size=5, fp16=use_fp16)  # Beam search
            except Exception:  # Fallback to array if file path fails again
                _, audio_data = wavfile.read(wav_path)  # Read WAV data
                audio_float = audio_data.astype(np.float32) / 32768.0  # Convert to float
                if audio_float.ndim > 1:  # Check channels
                    audio_float = audio_float.mean(axis=1)  # Mix to mono
                result = model.transcribe(audio_float, language='hi', task='transcribe',  # Force Hindi
                                          initial_prompt=hindi_prompt, temperature=0.2,  # Higher temp
                                          beam_size=5, fp16=use_fp16)  # Beam search
            detected_lang = result.get('language', 'hi')  # Update language
            transcribed_text = result.get('text', '').strip()  # Update transcription

        transcribed_text = post_process_transcription(transcribed_text)  # Clean the output
        return transcribed_text, detected_lang  # Return cleaned text and language code

    except Exception as e:  # Catch any unexpected transcription error
        print(f'[SIO STT] Transcription error: {e}')  # Log error
        return None, None  # Return None on failure

# ── Temporary file cleanup ────────────────────────────────────────────────────── #
def remove_temporary_file(wav_path):  # Define function to delete temporary WAV file
    if wav_path and os.path.exists(wav_path):  # Check if file path exists on disk
        try:  # Try deleting the file
            os.remove(wav_path)  # Delete the temporary WAV file
            print(f'[SIO STT] Deleted temp file: {wav_path}')  # Confirm deletion
        except Exception as e:  # Catch deletion errors
            print(f'[SIO STT] Could not delete temp file: {e}')  # Log warning

# ── Standalone test ───────────────────────────────────────────────────────────── #
def main():  # Define main function to test Module 1 standalone
    print('=' * 65)  # Print banner top border
    print('  SIO — Module 1: Speech to Text (VAD + Noise Reduction)')  # Print title
    print('=' * 65)  # Print banner bottom border
    model = load_whisper_model()  # Load Whisper model
    if model is None:  # Check if model failed to load
        print('[SIO STT] Cannot continue without Whisper model.')  # Error message
        return  # Exit function
    temp_wav = record_audio_to_wav()  # Record audio with VAD
    if temp_wav is None:  # Check if recording failed
        print('[SIO STT] Recording failed. Check microphone.')  # Error message
        return  # Exit function
    try:  # Try transcribing the recorded audio
        text, lang = transcribe_audio(model, temp_wav)  # Transcribe speech
        if text:  # Check if transcription returned text
            lang_label = 'Hindi' if lang == 'hi' else ('English' if lang == 'en' else lang)  # Friendly label
            print('-' * 65)  # Print section divider
            print(f'  Language : {lang_label} ({lang})')  # Print language
            print(f'  Text     : {text}')  # Print transcription
            print('-' * 65)  # Print section divider
        else:  # If transcription failed
            print('[SIO STT] No speech detected. Please try again.')  # Inform user
    finally:  # Always clean up temp file
        remove_temporary_file(temp_wav)  # Delete temp WAV file

if __name__ == '__main__':  # Check if run directly
    main()  # Run standalone test
