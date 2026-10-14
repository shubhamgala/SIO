# SIO Voice-First Banking Assistant - Module 1: Speech to Text (STT)              # File title comment
# This script records audio and transcribes Hindi, English, and Hinglish speech.  # Module description comment
# Features: dynamic active mic detection, continuous recording, audio normalization. # Features comment

import os  # Import os module for file and path operations
import re  # Import re module for regular expression text cleaning
import sys  # Import sys module for system configurations
import time  # Import time module for countdown display and pauses
import tempfile  # Import tempfile module for secure temporary file paths
import subprocess  # Import subprocess module to execute pip commands

# Reconfigure stdout and stderr to UTF-8 so Hindi characters print cleanly       # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if stdout supports reconfiguration
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout encoding to UTF-8
if hasattr(sys.stderr, 'reconfigure'):  # Check if stderr supports reconfiguration
    sys.stderr.reconfigure(encoding='utf-8')  # Set stderr encoding to UTF-8

# List of required external packages for STT module                              # Description comment
REQUIRED_PACKAGES = [  # Define required packages
    ("openai-whisper", "whisper"),  # OpenAI Whisper for speech recognition
    ("sounddevice", "sounddevice"),  # sounddevice for audio capture
    ("scipy", "scipy"),  # scipy for WAV processing and signal filtering
    ("numpy", "numpy"),  # numpy for audio array operations
    ("imageio-ffmpeg", "imageio_ffmpeg"),  # Bundled ffmpeg binaries
]  # End of required packages list

def ensure_dependencies():  # Define helper function to verify packages
    for pip_name, import_name in REQUIRED_PACKAGES:  # Loop through packages
        try:  # Try importing package
            __import__(import_name)  # Attempt import
        except ImportError:  # If package missing
            print(f"[SIO STT] Installing missing package '{pip_name}'...")  # Inform user
            try:  # Try pip install
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "--quiet"])  # Install
            except Exception as err:  # Catch installation error
                print(f"[SIO STT] Warning: Could not install '{pip_name}': {err}")  # Warn user

ensure_dependencies()  # Run package verification

import numpy as np  # Import numpy for numerical array processing
import sounddevice as sd  # Import sounddevice for microphone recording
from scipy.io import wavfile  # Import wavfile to read and write WAV audio
from scipy.signal import butter, sosfilt  # Import butter and sosfilt for audio filtering
import whisper  # Import Whisper library for speech transcription
import torch  # Import torch for GPU availability checks

# Configure ffmpeg path from imageio_ffmpeg if available                         # Description comment
def configure_ffmpeg():  # Define function to ensure ffmpeg is accessible
    try:  # Start try block
        import imageio_ffmpeg  # Import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()  # Get path to ffmpeg binary
        d = os.path.dirname(exe)  # Get directory
        if d not in os.environ.get("PATH", ""):  # Check if in PATH
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")  # Add to PATH
        std = os.path.join(d, "ffmpeg.exe")  # Standard executable name
        if not os.path.exists(std) and os.path.exists(exe):  # If standard name missing
            import shutil  # Import shutil
            shutil.copyfile(exe, std)  # Copy versioned binary to ffmpeg.exe
    except Exception as err:  # Catch configuration error
        print(f"[SIO STT] Note on ffmpeg setup: {err}")  # Print note

configure_ffmpeg()  # Run ffmpeg setup

# Configuration constants                                                        # Description comment
MODEL_NAME = "base"  # Whisper model name ('base' for fast balanced performance)
SAMPLE_RATE = 16000  # Standard audio sample rate expected by Whisper
DURATION = 5  # Recording duration in seconds
SUPPORTED_LANGUAGES = {"hi", "en"}  # Languages supported by SIO assistant

# Function to detect which microphone currently has an active audio signal        # Description comment
def get_best_microphone_device():  # Define function to auto-pick the best microphone
    try:  # Start try block
        devices = sd.query_devices()  # Retrieve all audio devices
        input_devs = [i for i, d in enumerate(devices) if d.get('max_input_channels', 0) > 0]  # Filter inputs
        if not input_devs:  # Check if no input device exists
            print("[SIO STT] Error: No microphone input found.", flush=True)  # Error log
            return None, "No Microphone"  # Return None
        default_in = sd.default.device[0]  # Get system default input device index
        # Prioritize system default input device and any Bluetooth headset device
        priority_devs = []  # List of prioritized device indices
        if default_in is not None and default_in in input_devs:  # Check if default is valid
            priority_devs.append(default_in)  # Test default device first
        for idx in input_devs:  # Loop through all input devices
            name_lower = devices[idx]['name'].lower()  # Get lowercase name
            if ("buds" in name_lower or "headset" in name_lower or "bluetooth" in name_lower) and idx not in priority_devs:  # Check headset
                priority_devs.append(idx)  # Prioritize headset
        for idx in input_devs:  # Add remaining candidate input devices
            if idx not in priority_devs:  # Avoid duplicate entries
                priority_devs.append(idx)  # Append remaining device
        best_id = None  # Track best device id
        best_peak = -1.0  # Track highest peak signal
        # Fast 0.15s signal check on candidate inputs to find which one is hearing voice # Comment
        for idx in priority_devs[:3]:  # Check top 3 candidate devices quickly
            dev = devices[idx]  # Get device info
            try:  # Try recording test burst
                snippet = sd.rec(int(0.15 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=idx)  # Record
                sd.wait()  # Wait for burst
                peak = float(np.max(np.abs(snippet)))  # Calculate peak amplitude
                if peak > best_peak:  # If louder than previous best
                    best_peak = peak  # Update peak
                    best_id = idx  # Update device id
                if peak > 0.005:  # If strong voice signal is found immediately
                    break  # Break early to start recording without delay
            except Exception:  # If device fails to open
                continue  # Skip to next device
        if best_id is not None and best_peak > 0.001:  # If an active device with real sound was found
            name = devices[best_id]['name']  # Get device name
            print(f"[SIO STT] Active microphone: [{best_id}] {name} (Signal: {best_peak:.5f})", flush=True)  # Log selection
            return best_id, name  # Return chosen device
        # Fallback to system default input device if signal test was inconclusive  # Comment
        if default_in is None or default_in < 0:  # If default device index invalid
            default_in = input_devs[0]  # Take first available device
        default_name = devices[default_in]['name']  # Get default name
        print(f"[SIO STT] Using default microphone: [{default_in}] {default_name}", flush=True)  # Log default
        if best_peak < 0.0005:  # If audio signal is close to digital silence
            print("[SIO STT] ⚠️ Notice: Microphone signal is very low. If muted, check your F4 key or Windows Sound Settings.", flush=True)  # Warn
        return default_in, default_name  # Return default device
    except Exception as e:  # Catch any device query error
        print(f"[SIO STT] Warning during device check: {e}", flush=True)  # Log warning
        return None, "Default"  # Return None fallback

# Function to remove low-frequency rumble and fan noise                           # Description comment
def apply_audio_filter(data, cutoff=80.0, fs=16000.0):  # Define high-pass filter function
    try:  # Start try block
        sos = butter(4, cutoff, 'hp', fs=fs, output='sos')  # Design 4th order highpass filter at 80Hz
        return sosfilt(sos, data).astype(np.float32)  # Filter audio and return float32
    except Exception:  # If filtering fails
        return data  # Return original unfiltered audio

# Function to record audio for 5 seconds and save to a temporary WAV file        # Description comment
def record_audio_to_wav():  # Define audio recording function
    device_id, dev_name = get_best_microphone_device()  # Find active microphone device
    temp_wav_path = None  # Initialize temp file path
    try:  # Start try block
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:  # Create temporary WAV file
            temp_wav_path = tmp.name  # Store file path
        print("[SIO STT] Get ready to speak in:", flush=True)  # Prompt user immediately
        for count in range(3, 0, -1):  # 3-second countdown loop
            print(f"[SIO STT]   {count}...", flush=True)  # Display countdown number immediately
            time.sleep(1)  # Pause 1 second
        print(f"[SIO STT] >>> SPEAK NOW! (Hindi / English / Hinglish for {DURATION}s) <<<", flush=True)  # Prompt to speak
        raw_audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=device_id)  # Record
        sd.wait()  # Wait for recording to complete
        print("[SIO STT] Recording finished! Processing audio...", flush=True)  # Inform recording done
        mono = raw_audio.flatten()  # Flatten into 1D audio array
        mono = apply_audio_filter(mono)  # Apply highpass filter to cut fan and room rumble
        peak = float(np.max(np.abs(mono)))  # Measure peak audio volume
        if peak > 0.0005:  # Check if real audio signal exists
            mono = (mono / peak) * 0.95  # Normalize volume to 95% full scale for clear transcription
        else:  # If audio is virtually silent
            print("[SIO STT] ⚠️ Audio signal was extremely faint. Please check microphone volume.", flush=True)  # Warn
        int16_audio = (mono * 32767.0).astype(np.int16)  # Convert float32 samples to 16-bit PCM format
        wavfile.write(temp_wav_path, SAMPLE_RATE, int16_audio)  # Save audio to temporary WAV file
        print(f"[SIO STT] Audio saved temporarily at: {temp_wav_path}", flush=True)  # Log saved file path
        return temp_wav_path  # Return WAV path
    except Exception as rec_err:  # Catch any recording error
        print(f"[SIO STT] Error during recording: {rec_err}")  # Log error
        if temp_wav_path and os.path.exists(temp_wav_path):  # Clean up partial file
            try:  # Try removing file
                os.remove(temp_wav_path)  # Delete
            except OSError:  # Ignore cleanup error
                pass  # Continue
        return None  # Return None on failure

# Function to load OpenAI Whisper speech recognition model                       # Description comment
def load_whisper_model():  # Define Whisper loader function
    try:  # Start try block
        print(f"[SIO STT] Loading OpenAI Whisper '{MODEL_NAME}' model...")  # Status message
        model = whisper.load_model(MODEL_NAME)  # Load model into memory
        print(f"[SIO STT] OpenAI Whisper '{MODEL_NAME}' model ready.")  # Confirmation message
        return model  # Return model instance
    except Exception as err:  # Catch loading error
        print(f"[SIO STT] Error loading Whisper model: {err}")  # Log error
        return None  # Return None on failure

# Function to clean Whisper hallucination artifacts from transcription text       # Description comment
def post_process_transcription(text):  # Define transcription post-processing function
    if not text:  # Check if text is empty
        return text  # Return empty text
    cleaned = text.strip()  # Strip outer whitespace
    # Remove leading English filler words inserted by Whisper                     # Comment
    cleaned = re.sub(r'^\s*(?:and|oh|um|uh|so|well|now|okay|ok|hmm|hey)\s+', '', cleaned, flags=re.IGNORECASE)  # Remove filler
    # Normalize Indian currency formatting Rs.5,000 -> 5000                      # Comment
    cleaned = re.sub(r'\bRs\.?\s*([\d,]+)', lambda m: m.group(1).replace(',', ''), cleaned, flags=re.IGNORECASE)  # Strip commas
    # Strip standalone comma formatting 5,000 -> 5000                             # Comment
    cleaned = re.sub(r'\b(\d{1,3}(?:,\d{2,3})+)\b', lambda m: m.group(0).replace(',', ''), cleaned)  # Strip commas
    # Strip trailing currency noise artifacts e.g. "to Rs."                       # Comment
    cleaned = re.sub(r'\s+(?:to\s+)?(?:Rs\.?|INR)\s*$', '', cleaned, flags=re.IGNORECASE)  # Remove trailing noise
    return cleaned.strip()  # Return cleaned text

# Function to transcribe audio using Whisper in Hindi, English, or Hinglish      # Description comment
def transcribe_audio(model, wav_path):  # Define transcription function
    try:  # Start try block
        print("[SIO STT] Transcribing audio (Hindi / English / Hinglish)...")  # Inform user
        use_fp16 = torch.cuda.is_available()  # Check if CUDA GPU is available for FP16
        multilingual_prompt = (  # Multilingual banking context prompt to guide Whisper
            "नमस्ते, मेरा बैलेंस कितना है? राजू को 500 रुपये भेजो। "  # Hindi banking context
            "Hello, what is my account balance? Send 500 rupees to Raju. Mobile recharge."  # English context
        )  # End of prompt string
        try:  # Try transcribing directly with file path
            result = model.transcribe(wav_path, task="transcribe", initial_prompt=multilingual_prompt, temperature=0.0, beam_size=5, fp16=use_fp16)  # Transcribe
        except Exception:  # If file path fails fallback to audio array
            _, data = wavfile.read(wav_path)  # Read WAV file
            float_data = data.astype(np.float32) / 32768.0  # Normalize to float32
            if float_data.ndim > 1:  # Check if stereo
                float_data = float_data.mean(axis=1)  # Mix to mono
            result = model.transcribe(float_data, task="transcribe", initial_prompt=multilingual_prompt, temperature=0.0, beam_size=5, fp16=use_fp16)  # Transcribe
        detected_lang = result.get("language", "unknown")  # Retrieve detected language
        transcribed_text = result.get("text", "").strip()  # Retrieve transcribed text
        # If an unexpected language detected fallback to forcing Hindi pass       # Comment
        if detected_lang not in SUPPORTED_LANGUAGES and len(transcribed_text) < 3:  # Check condition
            print(f"[SIO STT] Detected '{detected_lang}' — re-trying in Hindi...")  # Inform user
            hindi_prompt = "नमस्ते, मेरा खाता, बैलेंस, पैसे ट्रांसफर, रिचार्ज, बिल।"  # Hindi prompt
            result = model.transcribe(wav_path, language="hi", task="transcribe", initial_prompt=hindi_prompt, temperature=0.2, fp16=use_fp16)  # Retry
            detected_lang = result.get("language", "hi")  # Update language
            transcribed_text = result.get("text", "").strip()  # Update text
        transcribed_text = post_process_transcription(transcribed_text)  # Apply post-processing cleaner
        return transcribed_text, detected_lang  # Return final text and language
    except Exception as err:  # Catch any transcription error
        print(f"[SIO STT] Error during transcription: {err}")  # Log error
        return None, None  # Return None on failure

# Function to remove temporary WAV file                                          # Description comment
def remove_temporary_file(wav_path):  # Define cleanup function
    if wav_path and os.path.exists(wav_path):  # Check if file exists
        try:  # Start try block
            os.remove(wav_path)  # Delete file
            print(f"[SIO STT] Deleted temporary file: {wav_path}")  # Log deletion
        except Exception as err:  # Catch deletion error
            print(f"[SIO STT] Warning: Could not delete temporary file: {err}")  # Warn user
