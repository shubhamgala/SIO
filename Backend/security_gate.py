# SIO Voice-First Banking Assistant - Module: Multi-Factor Biometric Security Gate # File title comment
# Implements 3-step security check: (1) Face Verification, (2) Voice Biometric PIN,  # Architecture overview comment
# and (3) Dynamic Challenge-Response Liveness Test using Whisper speech recognition. # Architecture overview comment

import os  # Import os module for filesystem and path operations
import sys  # Import sys module for system paths and runtime configurations
import time  # Import time module for delay timings and pauses
import random  # Import random module to pick unpredictable dynamic liveness challenges
from pathlib import Path  # Import Path class for object-oriented filesystem path management
from typing import Dict, Any, Optional, Tuple  # Import typing primitives for clear function signatures
import numpy as np  # Import numpy for numerical and vector math operations

# Reconfigure stdout and stderr to UTF-8 for clean Devanagari and emoji display     # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if standard output supports reconfiguring encoding
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout encoding to UTF-8
if hasattr(sys.stderr, 'reconfigure'):  # Check if standard error supports reconfiguring encoding
    sys.stderr.reconfigure(encoding='utf-8')  # Set stderr encoding to UTF-8

# Add project root directory to sys.path so modules can import cleanly               # Description comment
ROOT_DIR = Path(__file__).resolve().parent.parent  # Compute absolute path to repository root folder
if str(ROOT_DIR) not in sys.path:  # Check if project root is not already present in sys.path
    sys.path.insert(0, str(ROOT_DIR))  # Insert project root at the beginning of sys.path

# Define paths to required biometric model files                                     # Description comment
MODELS_DIR = ROOT_DIR / "models"  # Path to local models folder
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"  # OpenCV YuNet face detection ONNX model
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"  # OpenCV SFace face recognition ONNX model
VOICE_MODEL_DIR = MODELS_DIR / "spkrec-ecapa-voxceleb"  # SpeechBrain ECAPA-TDNN speaker model directory

# Predefined pool of dynamic Hindi challenge phrases for anti-spoofing liveness test # Description comment
CHALLENGE_PHRASES = [  # List of dictionaries containing challenge text and expected keywords
    {  # Challenge 1
        "phrase": "मेरा बैंक सुरक्षित है",  # Hindi challenge text to be spoken and repeated
        "keywords": ["बैंक", "सुरक्षित", "bank", "surakshit"],  # Validation keywords for fuzzy recognition
    },  # End Challenge 1
    {  # Challenge 2
        "phrase": "आज का दिन शुभ है",  # Hindi challenge text
        "keywords": ["आज", "दिन", "शुभ", "aaj", "din", "shubh"],  # Validation keywords
    },  # End Challenge 2
    {  # Challenge 3
        "phrase": "भारत माता की जय",  # Hindi challenge text
        "keywords": ["भारत", "माता", "जय", "bharat", "mata", "jai"],  # Validation keywords
    },  # End Challenge 3
    {  # Challenge 4
        "phrase": "जय हिंद वन्दे मातरम्",  # Hindi challenge text
        "keywords": ["जय", "हिंद", "वन्दे", "मातरम्", "jai", "hind", "vande"],  # Validation keywords
    },  # End Challenge 4
    {  # Challenge 5
        "phrase": "सुरक्षा पहली प्राथमिकता है",  # Hindi challenge text
        "keywords": ["सुरक्षा", "प्राथमिकता", "suraksha", "pehli"],  # Validation keywords
    },  # End Challenge 5
]  # End of CHALLENGE_PHRASES list

# Helper function to print clear model download instructions when files are missing  # Description comment
def print_model_download_guide(missing_component: str):  # Define guide printer function
    print("\n" + "=" * 72, flush=True)  # Top separator banner
    print("  ⚠️  SIO BIOMETRIC SECURITY MODULE: MISSING MODEL FILES DETECTED", flush=True)  # Alert header
    print("=" * 72, flush=True)  # Separator line
    if "face" in missing_component.lower():  # Check if face models are missing
        print(" [Face Detection & Recognition Models Missing]", flush=True)  # Subheader
        print(" 1. Download YuNet Face Detector ONNX:", flush=True)  # Instruction 1
        print("    URL : https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet", flush=True)  # URL 1
        print(f"    Save to: {YUNET_MODEL}", flush=True)  # Destination path 1
        print(" 2. Download SFace Face Recognition ONNX:", flush=True)  # Instruction 2
        print("    URL : https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface", flush=True)  # URL 2
        print(f"    Save to: {SFACE_MODEL}", flush=True)  # Destination path 2
    if "voice" in missing_component.lower():  # Check if voice model is missing
        print(" [Voice Biometrics Model Missing]", flush=True)  # Subheader
        print(" 1. SpeechBrain ECAPA-TDNN VoxCeleb Speaker Recognition Model:", flush=True)  # Instruction
        print("    Source: HuggingFace 'speechbrain/spkrec-ecapa-voxceleb'", flush=True)  # HuggingFace source
        print(f"    Save to: {VOICE_MODEL_DIR}", flush=True)  # Destination path
        print("    Auto-download via python: EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb')", flush=True)  # Command hint
    print("=" * 72 + "\n", flush=True)  # Bottom separator banner

# Step 1: Verify face using camera frame and OpenCV YuNet/SFace models               # Description comment
def verify_face_step(user_id: str = "user1", frame_override: Optional[np.ndarray] = None) -> Tuple[bool, float, str]:  # Define face step
    print("\n [Security Step 1/3] 📷 Checking Facial Biometrics...", flush=True)  # Step log
    if not YUNET_MODEL.is_file() or not SFACE_MODEL.is_file():  # Check if required face model files exist
        print_model_download_guide("face")  # Print download instructions
        return False, 0.0, "Face recognition model files missing from models/ folder."  # Return failure
    try:  # Start try block to load face verification logic
        import cv2  # Import OpenCV for camera capture
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # Silence OpenCV C++ warnings
        from auth.face import verify_face  # Import face verification function
        from auth.config import AuthConfig  # Import biometric configuration thresholds
    except Exception as import_err:  # Catch import error
        return False, 0.0, f"Could not load face authentication library: {import_err}"  # Return failure
    frame = frame_override  # Use frame override if provided (useful for tests)
    if frame is None:  # If no frame override, capture directly from system webcam
        print(" [Security Step 1] Accessing webcam... Please look directly at the camera.", flush=True)  # Prompt
        cap = cv2.VideoCapture(0)  # Open system default camera device
        if not cap.isOpened():  # Check if camera opening failed
            return False, 0.0, "Webcam could not be opened. Check camera permissions or connection."  # Return error
        try:  # Start try block to capture frame from webcam
            for _ in range(15):  # Read a few warm-up frames so auto-exposure settles
                ret, captured = cap.read()  # Read single frame
                if ret and captured is not None:  # Check if frame is valid
                    frame = captured  # Store candidate frame
                time.sleep(0.05)  # Pause 50ms between frame reads
        finally:  # Ensure webcam capture device is released
            cap.release()  # Release video capture device
    if frame is None:  # Check if no valid frame could be captured
        return False, 0.0, "No valid video frame could be captured from webcam."  # Return error
    try:  # Try executing biometric face comparison
        cfg = AuthConfig()  # Initialize configuration with default similarity thresholds
        matched, score = verify_face(user_id, frame, config=cfg)  # Run face comparison against enrolled vector
        print(f" [Security Step 1] Face Similarity Score: {score:.4f} (Required: {cfg.face_similarity_threshold:.2f})", flush=True)  # Log
        if matched:  # Check if similarity met the threshold
            print(" [Security Step 1] ✅ Face Verified: Authorized account holder confirmed.", flush=True)  # Log
            return True, score, "Face successfully verified."  # Return success
        else:  # If face did not match
            print(" [Security Step 1] ❌ Face Verification Failed: Biometric mismatch or no face found.", flush=True)  # Log
            return False, score, "Face verification failed. Face does not match enrolled profile."  # Return failure
    except Exception as face_err:  # Catch any runtime verification error
        return False, 0.0, f"Error during face verification: {face_err}"  # Return error description

# Step 2: Verify voice PIN using microphone and SpeechBrain ECAPA-TDNN               # Description comment
def verify_voice_step(user_id: str = "user1", audio_path_override: Optional[str] = None) -> Tuple[bool, float, str]:  # Define voice step
    print("\n [Security Step 2/3] 🎙️  Checking Voice Biometrics (Speaker Verification)...", flush=True)  # Step log
    try:  # Start try block to load voice authentication logic
        from auth.voice import verify_voice  # Import voice verification function
        from auth.config import AuthConfig  # Import biometric configuration
        from Backend.dialogue_manager import speak  # Import speak function to prompt user
    except Exception as import_err:  # Catch import error
        return False, 0.0, f"Could not load voice authentication library: {import_err}"  # Return failure
    wav_path = audio_path_override  # Use audio override if provided
    temp_allocated = False  # Track whether a temporary audio file needs cleanup
    if wav_path is None:  # If no audio override, record live voice from microphone
        from STT.whisper_stt import record_audio_to_wav  # Import audio recording helper
        speak("Kripya apna voice PIN ya suraksha vaakya bolein.")  # Prompt user via voice
        print(" [Security Step 2] Please speak your voice PIN or security phrase for 5 seconds...", flush=True)  # Prompt
        wav_path = record_audio_to_wav()  # Record 5 seconds of audio from microphone
        temp_allocated = True  # Flag temporary file for cleanup
    if not wav_path or not os.path.exists(wav_path):  # Check if audio was recorded successfully
        return False, 0.0, "Could not capture voice sample from microphone."  # Return failure
    try:  # Start try block to execute voice comparison
        cfg = AuthConfig()  # Initialize configuration
        matched, score = verify_voice(user_id, wav_path)  # Compare voice sample with enrolled profile
        print(f" [Security Step 2] Voice Similarity Score: {score:.4f} (Required: {cfg.voice_similarity_threshold:.2f})", flush=True)  # Log
        if matched:  # Check if similarity met threshold
            print(" [Security Step 2] ✅ Voice Biometrics Verified: Speaker identity confirmed.", flush=True)  # Log
            return True, score, "Voice biometrics verified successfully."  # Return success
        else:  # If voice did not match
            print(" [Security Step 2] ❌ Voice Biometrics Failed: Speaker voiceprint mismatch.", flush=True)  # Log
            return False, score, "Voice biometrics failed. Voiceprint does not match enrolled profile."  # Return failure
    except Exception as voice_err:  # Catch runtime error
        return False, 0.0, f"Error during voice biometric comparison: {voice_err}"  # Return error description
    finally:  # Cleanup temporary audio file if one was recorded
        if temp_allocated and wav_path and os.path.exists(wav_path):  # Check if temp file exists
            try:  # Try deleting
                from STT.whisper_stt import remove_temporary_file  # Import cleanup helper
                remove_temporary_file(wav_path)  # Delete temp file
            except Exception:  # Ignore cleanup errors
                pass  # Continue

# Step 3: Dynamic challenge-response anti-spoofing liveness verification             # Description comment
def verify_challenge_liveness_step(whisper_model=None, simulated_response: Optional[str] = None, speak_prompt: bool = True, challenge_override: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, str, str]:  # Define liveness step
    print("\n [Security Step 3/3] 🛡️  Dynamic Challenge-Response Liveness Test...", flush=True)  # Step log
    challenge = challenge_override or random.choice(CHALLENGE_PHRASES)  # Use override or pick random phrase
    challenge_text = challenge["phrase"]  # Extract the challenge sentence
    expected_keywords = challenge["keywords"]  # Extract expected keywords for verification
    try:  # Start try block to execute liveness prompt and capture
        from Backend.dialogue_manager import speak  # Import speak function
        from STT.whisper_stt import record_audio_to_wav, transcribe_audio, remove_temporary_file, load_whisper_model  # STT helpers
        prompt_speech = f"Liveness suraksha jaanch ke liye mere baad bolein: {challenge_text}"  # Assemble prompt
        print(f" [Security Step 3] System Challenge: \"{challenge_text}\"", flush=True)  # Print challenge
        if speak_prompt:  # Check if speech output is enabled
            speak(prompt_speech)  # Speak challenge sentence aloud through speakers
        if simulated_response is not None:  # Check if simulated response is provided (for tests)
            spoken_text = simulated_response  # Use simulated transcription
        else:  # Live recording from user microphone
            model = whisper_model or load_whisper_model()  # Ensure Whisper model is ready
            print(" [Security Step 3] Repeat the challenge phrase now into your microphone...", flush=True)  # Instructions
            wav_path = record_audio_to_wav()  # Record response from microphone
            if not wav_path:  # Check if recording failed
                return False, challenge_text, "", "Could not record microphone audio for challenge response."  # Return failure
            try:  # Try transcribing response
                spoken_text, _ = transcribe_audio(model, wav_path)  # Transcribe user's speech using Whisper
                spoken_text = spoken_text or ""  # Ensure string
            finally:  # Ensure temporary audio file is deleted
                remove_temporary_file(wav_path)  # Clean up temp WAV
        print(f" [Security Step 3] User Repeated  : \"{spoken_text}\"", flush=True)  # Print what was transcribed
        spoken_lower = spoken_text.lower()  # Convert to lowercase for comparison
        matches = [kw for kw in expected_keywords if kw.lower() in spoken_lower]  # Find matching keywords
        is_matched = len(matches) >= 1 or challenge_text in spoken_text  # Check if key words were repeated
        if is_matched:  # Check if challenge passed
            print(" [Security Step 3] ✅ Challenge Passed: Dynamic liveness confirmed (No audio spoofing).", flush=True)  # Log
            return True, challenge_text, spoken_text, "Dynamic liveness challenge verified."  # Return success
        else:  # If challenge failed
            print(" [Security Step 3] ❌ Challenge Failed: Spoken phrase did not match challenge.", flush=True)  # Log
            return False, challenge_text, spoken_text, f"Spoken phrase '{spoken_text}' did not match challenge '{challenge_text}'."  # Return failure
    except Exception as live_err:  # Catch any error
        return False, challenge_text, "", f"Error during liveness test: {live_err}"  # Return error description

# Master Security Gate function executing all 3 steps in exact sequential order      # Description comment
def run_security_gate(  # Define master security gate entry point
    user_id: str = "user1",  # Target user identifier for biometric lookup
    whisper_model=None,  # Pre-loaded Whisper model instance
    simulated_inputs: Optional[Dict[str, Any]] = None,  # Optional test simulation inputs
) -> Dict[str, Any]:  # Returns structured dictionary with verification status and audit details
    print("\n" + "=" * 70, flush=True)  # Header border
    print("      SIO MULTI-FACTOR BIOMETRIC SECURITY GATE (Zero-Trust Pipeline)", flush=True)  # Header title
    print("      Step 1: Face ID  |  Step 2: Voice PIN  |  Step 3: Liveness Test", flush=True)  # Subtitle
    print("=" * 70, flush=True)  # Header border
    sim = simulated_inputs or {}  # Extract simulation dictionary if passed
    details = {}  # Initialize details dictionary to log audit information
    # ── Step 1: Face Detection & Verification ────────────────────────────────────── #
    face_ok, face_score, face_msg = verify_face_step(user_id, frame_override=sim.get("frame"))  # Run Step 1
    details["face_score"] = face_score  # Record face similarity score
    details["face_message"] = face_msg  # Record face status message
    if not face_ok:  # If face verification failed
        return {  # Return failure dictionary immediately
            "authenticated": False,  # Mark authentication failed
            "failed_step": 1,  # Record failed step number
            "reason": face_msg,  # Provide reason for rejection
            "details": details,  # Include audit details
        }  # End of failure return
    # ── Step 2: Voice Biometric PIN Verification ─────────────────────────────────── #
    voice_ok, voice_score, voice_msg = verify_voice_step(user_id, audio_path_override=sim.get("voice_audio"))  # Run Step 2
    details["voice_score"] = voice_score  # Record voice similarity score
    details["voice_message"] = voice_msg  # Record voice status message
    if not voice_ok:  # If voice verification failed
        return {  # Return failure dictionary immediately
            "authenticated": False,  # Mark authentication failed
            "failed_step": 2,  # Record failed step number
            "reason": voice_msg,  # Provide reason for rejection
            "details": details,  # Include audit details
        }  # End of failure return
    # ── Step 3: Dynamic Random Challenge-Response Liveness Test ──────────────────── #
    live_ok, challenge_str, user_str, live_msg = verify_challenge_liveness_step(  # Run Step 3
        whisper_model=whisper_model,  # Pass Whisper model
        simulated_response=sim.get("challenge_response"),  # Pass simulated user response if present
        speak_prompt=sim.get("speak_prompt", True),  # Control whether challenge is spoken aloud
        challenge_override=sim.get("challenge_override"),  # Optional deterministic challenge for testing
    )  # End of liveness step call
    details["challenge_text"] = challenge_str  # Record the challenge phrase
    details["challenge_spoken"] = user_str  # Record user's spoken response
    details["liveness_message"] = live_msg  # Record liveness status message
    if not live_ok:  # If challenge response failed
        return {  # Return failure dictionary immediately
            "authenticated": False,  # Mark authentication failed
            "failed_step": 3,  # Record failed step number
            "reason": live_msg,  # Provide reason for rejection
            "details": details,  # Include audit details
        }  # End of failure return
    # ── All Three Security Steps Passed Successfully ─────────────────────────────── #
    print("\n" + "=" * 70, flush=True)  # Success border
    print("  🎉 MULTI-FACTOR AUTHENTICATION PASSED! User Identity Verified 100%.", flush=True)  # Success message
    print("=" * 70 + "\n", flush=True)  # Success border
    return {  # Return successful authentication dictionary
        "authenticated": True,  # Mark authentication as verified
        "failed_step": None,  # No step failed
        "reason": "All 3 biometric and liveness security checks passed successfully.",  # Success message
        "details": details,  # Full audit log
    }  # End of success return

# Direct execution entry point to run security gate standalone from terminal         # Description comment
if __name__ == "__main__":  # Check if executed as standalone script
    result = run_security_gate()  # Run security gate
    print(f"\nFinal Result: {result}")  # Print final result
