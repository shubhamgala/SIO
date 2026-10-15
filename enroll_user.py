# SIO Voice-First Banking Assistant - User Biometric Enrollment Script            # File title comment
# Enrolls new elderly users by capturing their face via webcam and voice sample   # Script purpose comment
# Saves face embedding (128-d) and voice embedding (192-d) into secure storage.   # Architecture overview comment

import os  # Import os module for directory and file path operations
import sys  # Import sys module for system paths and runtime configurations
import time  # Import time module for countdown delays and frame settling
from pathlib import Path  # Import Path class for filesystem path manipulation
import numpy as np  # Import numpy for array manipulation and vector calculations

# Reconfigure standard output streams to UTF-8 for displaying Hindi and symbols   # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if standard output supports reconfiguring encoding
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout encoding to UTF-8
if hasattr(sys.stderr, 'reconfigure'):  # Check if standard error supports reconfiguring encoding
    sys.stderr.reconfigure(encoding='utf-8')  # Set stderr encoding to UTF-8

# Add project root directory to sys.path so modules can import cleanly           # Description comment
ROOT_DIR = Path(__file__).resolve().parent  # Compute absolute path to repository root folder
if str(ROOT_DIR) not in sys.path:  # Check if project root is not already present in sys.path
    sys.path.insert(0, str(ROOT_DIR))  # Insert project root at the beginning of sys.path

# Define paths to biometric models                                               # Description comment
MODELS_DIR = ROOT_DIR / "models"  # Path to local models directory
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"  # OpenCV YuNet face detection ONNX model
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"  # OpenCV SFace face recognition ONNX model
VOICE_MODEL_DIR = MODELS_DIR / "spkrec-ecapa-voxceleb"  # SpeechBrain ECAPA-TDNN speaker model directory

# Function to check model availability and print download guide if missing       # Description comment
def verify_model_prerequisites() -> bool:  # Define model verification function
    missing_items = []  # Initialize list of missing components
    if not YUNET_MODEL.is_file():  # Check if YuNet model file exists
        missing_items.append("YuNet Face Detection Model (face_detection_yunet_2023mar.onnx)")  # Add to missing
    if not SFACE_MODEL.is_file():  # Check if SFace model file exists
        missing_items.append("SFace Face Recognition Model (face_recognition_sface_2021dec.onnx)")  # Add to missing
    if not VOICE_MODEL_DIR.exists():  # Check if SpeechBrain voice model directory exists
        missing_items.append("SpeechBrain ECAPA-TDNN Voice Model (models/spkrec-ecapa-voxceleb)")  # Add to missing
    if missing_items:  # Check if any model files are missing
        print("\n" + "=" * 75, flush=True)  # Top banner border
        print("  ⚠️  CANNOT PROCEED: BIOMETRIC PRETRAINED MODEL FILES NOT FOUND", flush=True)  # Alert header
        print("=" * 75, flush=True)  # Border
        print(" The following required model files are missing from the project:\n", flush=True)  # Notice
        for item in missing_items:  # Loop through missing components
            print(f"  ❌ Missing: {item}", flush=True)  # Print missing item
        print("\n" + "-" * 75, flush=True)  # Divider
        print(" [DOWNLOAD INSTRUCTIONS]", flush=True)  # Instructions header
        print(" 1. YuNet Face Detection ONNX:", flush=True)  # YuNet info
        print("    Download from : https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet", flush=True)  # URL
        print(f"    Save to       : {YUNET_MODEL}", flush=True)  # Destination
        print(" 2. SFace Face Recognition ONNX:", flush=True)  # SFace info
        print("    Download from : https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface", flush=True)  # URL
        print(f"    Save to       : {SFACE_MODEL}", flush=True)  # Destination
        print(" 3. SpeechBrain ECAPA-TDNN Voice Biometrics:", flush=True)  # Voice info
        print("    HuggingFace   : https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb", flush=True)  # URL
        print(f"    Save to       : {VOICE_MODEL_DIR}", flush=True)  # Destination
        print("=" * 75 + "\n", flush=True)  # Bottom banner border
        return False  # Return False indicating prerequisites failed
    return True  # Return True if all models are present

# Function to enroll user face using live webcam                                 # Description comment
def capture_and_enroll_face(user_id: str) -> bool:  # Define face enrollment function
    print("\n" + "=" * 65, flush=True)  # Step border
    print("  [Step 1 of 2] FACE BIOMETRIC ENROLLMENT", flush=True)  # Step title
    print("=" * 65, flush=True)  # Step border
    try:  # Start try block
        import cv2  # Import OpenCV library
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # Silence verbose OpenCV C++ warnings
        from auth.face import enroll_face_from_frame  # Import frame enrollment helper function
        from Backend.dialogue_manager import speak  # Import speak helper
    except Exception as err:  # Catch import error
        print(f" [Error] Could not initialize OpenCV or face module: {err}", flush=True)  # Error log
        return False  # Return failure
    speak("Kripya camera ki taraf dekhein. Aapka chehra scan ho raha hai.")  # Speak Hindi instruction
    print(" Opening webcam... Please position your face in the center of the camera.", flush=True)  # Instructions
    cap = cv2.VideoCapture(0)  # Open system default camera device
    if not cap.isOpened():  # Check if camera opened successfully
        print(" [Error] Could not access webcam. Please check camera connection or permissions.", flush=True)  # Error
        return False  # Return failure
    print(" Preparing camera in: 3...", flush=True)  # Countdown 3
    time.sleep(1)  # Wait 1s
    print(" Preparing camera in: 2...", flush=True)  # Countdown 2
    time.sleep(1)  # Wait 1s
    print(" Preparing camera in: 1...", flush=True)  # Countdown 1
    time.sleep(1)  # Wait 1s
    print(" 📸 Capturing face sample...", flush=True)  # Capture prompt
    enrolled = False  # Track face enrollment status
    try:  # Start try block for camera capture
        for attempt in range(40):  # Check webcam frames for up to 2 seconds
            ret, frame = cap.read()  # Capture single frame
            if ret and frame is not None:  # Check frame validity
                if enroll_face_from_frame(user_id, frame):  # Extract face embedding and save
                    enrolled = True  # Mark enrollment successful
                    break  # Stop search once enrolled
            time.sleep(0.05)  # Wait 50ms between frame reads
    finally:  # Ensure webcam capture device is released
        cap.release()  # Release video capture device
    if enrolled:  # Check if face was successfully enrolled
        print(f" ✅ Face Biometric Enrolled Successfully for user '{user_id}'!", flush=True)  # Success log
        return True  # Return success
    else:  # If no face could be detected in webcam frames
        print(" ❌ Could not detect a clear face. Ensure good lighting and look straight into camera.", flush=True)  # Error
        return False  # Return failure

# Function to enroll user voice using microphone                                 # Description comment
def capture_and_enroll_voice(user_id: str) -> bool:  # Define voice enrollment function
    print("\n" + "=" * 65, flush=True)  # Step border
    print("  [Step 2 of 2] VOICE BIOMETRIC ENROLLMENT", flush=True)  # Step title
    print("=" * 65, flush=True)  # Step border
    try:  # Start try block
        from auth.voice import enroll_voice  # Import voice enrollment function
        from STT.whisper_stt import record_audio_to_wav, remove_temporary_file  # Import audio helpers
        from Backend.dialogue_manager import speak  # Import speak helper
    except Exception as err:  # Catch import error
        print(f" [Error] Could not initialize voice module: {err}", flush=True)  # Error log
        return False  # Return failure
    speak("Ab microphone mein apna naam aur suraksha vaakya bolein.")  # Speak Hindi instruction
    print(" You will now record a 5-second voice sample to create your authentic voiceprint.", flush=True)  # Instructions
    print(" For example, say: 'Mera naam Shubham hai aur yeh mera suraksha voice PIN hai.'", flush=True)  # Example
    wav_path = record_audio_to_wav()  # Record 5 seconds of audio from microphone
    if not wav_path or not os.path.exists(wav_path):  # Check if recording failed
        print(" [Error] Audio recording failed. Check microphone connection.", flush=True)  # Error log
        return False  # Return failure
    try:  # Start try block for voice extraction and saving
        print(" Extracting 192-dimensional ECAPA-TDNN speaker embedding...", flush=True)  # Progress log
        success = enroll_voice(user_id, [wav_path])  # Extract and save voice embedding
        if success:  # Check if enrollment succeeded
            print(f" ✅ Voice Biometric Enrolled Successfully for user '{user_id}'!", flush=True)  # Success log
            return True  # Return success
        else:  # If enrollment failed
            print(" ❌ Voice embedding extraction failed. Please try recording again.", flush=True)  # Error log
            return False  # Return failure
    except Exception as voice_err:  # Catch runtime error
        print(f" [Error] Failed during voice enrollment: {voice_err}", flush=True)  # Error log
        return False  # Return failure
    finally:  # Always clean up temporary audio file
        remove_temporary_file(wav_path)  # Delete temp WAV file from disk

# Master enrollment function coordinating full onboarding process                # Description comment
def enroll_elderly_user():  # Define master onboarding function
    print("\n" + "=" * 70, flush=True)  # Main banner border
    print("    SIO VOICE BANKING ASSISTANT — ELDERLY USER BIOMETRIC ENROLLMENT", flush=True)  # Main banner title
    print("    Registers Facial Profile (SFace) and Voiceprint (ECAPA-TDNN)", flush=True)  # Main banner subtitle
    print("=" * 70, flush=True)  # Main banner border
    if not verify_model_prerequisites():  # Verify that required model files exist
        return  # Terminate early if models are missing
    try:  # Prompt user for account identifier
        input_id = input("\nEnter User ID to enroll [Default: user1]: ").strip()  # Read user input
    except (EOFError, KeyboardInterrupt):  # Handle non-interactive or abort input
        input_id = ""  # Default empty
    user_id = input_id if input_id else "user1"  # Resolve user ID defaulting to user1
    print(f"\n[SIO Enrollment] Enrolling biometrics for account holder: '{user_id}'...", flush=True)  # Confirmation
    face_enrolled = capture_and_enroll_face(user_id)  # Execute Step 1: Face enrollment
    if not face_enrolled:  # Check if face enrollment failed
        print(f"\n❌ Face enrollment failed. Biometric registration for '{user_id}' was NOT completed.", flush=True)  # Log
        return  # Abort enrollment
    voice_enrolled = capture_and_enroll_voice(user_id)  # Execute Step 2: Voice enrollment
    if not voice_enrolled:  # Check if voice enrollment failed
        print(f"\n❌ Voice enrollment failed. Biometric registration for '{user_id}' was NOT completed.", flush=True)  # Log
        return  # Abort enrollment
    # Both face and voice biometrics enrolled successfully                        # Description comment
    embeddings_dir = ROOT_DIR / "data" / "embeddings"  # Path to saved embeddings directory
    print("\n" + "=" * 70, flush=True)  # Success banner border
    print(f"  🎉 ENROLLMENT COMPLETE: User '{user_id}' Successfully Registered!", flush=True)  # Success title
    print("=" * 70, flush=True)  # Success banner border
    print(f" • Face Profile Embedding  : {embeddings_dir / f'{user_id}_face.npy'}", flush=True)  # File path 1
    print(f" • Voice Profile Embedding : {embeddings_dir / f'{user_id}_voice.npy'}", flush=True)  # File path 2
    print(" • Security Protection     : Zero-Trust Multi-Factor Biometric Gate Ready ✓", flush=True)  # Status
    print("=" * 70 + "\n", flush=True)  # Footer border
    try:  # Announce completion using Hindi voice synthesis
        from Backend.dialogue_manager import speak  # Import speak helper
        speak("Aapka biometric enrollment safalta se poora ho gaya hai. Ab aap surakshit roop se SIO banking use kar sakte hain.")  # Hindi audio
    except Exception:  # Ignore voice error
        pass  # Continue

# Entry point for direct execution from command line                             # Description comment
if __name__ == "__main__":  # Check if run directly
    enroll_elderly_user()  # Run enrollment flow
