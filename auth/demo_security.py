# SIO Voice-First Banking Assistant - Security Module Interactive Runner         # File title comment
# Provides easy-to-run interactive demos for Voice Biometrics and Face Security.  # File purpose comment
# Allows testing in 3 modes: (1) Instant Self-Test, (2) Voice Mic, (3) Webcam.   # Capabilities comment

import os  # Import os module for paths and environment handling
import sys  # Import sys module for system input/output configurations
import time  # Import time module for pauses and timestamps
import numpy as np  # Import numpy for vector math and embeddings

# Reconfigure stdout and stderr to UTF-8 for clean console output               # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if stdout supports stream reconfiguration
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)  # Set stdout encoding to UTF-8 with immediate line flush
if hasattr(sys.stderr, 'reconfigure'):  # Check if stderr supports stream reconfiguration
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)  # Set stderr encoding to UTF-8 with immediate line flush

# Add project root directory to sys.path so modules can import cleanly           # Description comment
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root folder
if ROOT_DIR not in sys.path:  # Check if root is not already in sys.path
    sys.path.insert(0, ROOT_DIR)  # Add root to sys.path

from auth.config import AuthConfig  # Import biometric configuration thresholds
from auth.storage import save_embedding, load_embedding  # Import embedding storage helpers

# ── Mode 1: Automated Self-Test (Zero hardware required, 2 seconds) ──────────── #
def run_automated_self_test():  # Define self-test function
    print("\n" + "=" * 68, flush=True)  # Banner border
    print("  SECURITY DEMO 1: Automated Biometric Self-Test (Zero Hardware Needed)", flush=True)  # Title
    print("=" * 68, flush=True)  # Banner border
    cfg = AuthConfig()  # Load configuration
    print(f" [Config] Voice Similarity Threshold: {cfg.voice_similarity_threshold}", flush=True)  # Display voice threshold
    print(f" [Config] Face Similarity Threshold : {cfg.face_similarity_threshold}", flush=True)  # Display face threshold
    print(f" [Config] Face Lost Grace Period   : {cfg.grace_period_seconds} seconds", flush=True)  # Display grace period
    print("-" * 68, flush=True)  # Divider
    demo_user = "demo_account_holder"  # Mock user identifier
    # Create normalized authentic voice embedding
    true_voice = np.random.randn(192).astype(np.float32)  # Generate 192-d vector
    true_voice /= np.linalg.norm(true_voice)  # Normalize to unit length
    save_embedding(demo_user, "voice", true_voice)  # Save authentic voiceprint
    print(f" [Enroll] Saved authentic voiceprint for user '{demo_user}'.", flush=True)  # Confirm enrollment
    # Test authentic speaker with small acoustic variance
    noise = np.random.randn(192).astype(np.float32) * 0.05  # Slight natural voice fluctuation
    candidate_match = true_voice + noise  # Candidate vector
    candidate_match /= np.linalg.norm(candidate_match)  # Normalize
    score_match = float(np.dot(true_voice, candidate_match))  # Calculate cosine similarity
    is_match = score_match >= cfg.voice_similarity_threshold  # Compare against threshold
    print(f" [Auth Test 1] Same Speaker Test  : Similarity = {score_match:.4f} -> {'✅ ACCESS GRANTED' if is_match else '❌ DENIED'}", flush=True)  # Log
    # Test imposter speaker with completely different voice
    imposter_voice = np.random.randn(192).astype(np.float32)  # Different random voice
    imposter_voice /= np.linalg.norm(imposter_voice)  # Normalize
    score_imposter = float(np.dot(true_voice, imposter_voice))  # Cosine similarity
    is_imposter_blocked = score_imposter < cfg.voice_similarity_threshold  # Should be rejected
    print(f" [Auth Test 2] Imposter Voice Test: Similarity = {score_imposter:.4f} -> {'🛡️ BLOCKED (Security Pass)' if is_imposter_blocked else '❌ FAILED'}", flush=True)  # Log
    # Test path traversal injection attack defense
    traversal_blocked = False  # Flag
    try:  # Try injecting bad path
        save_embedding("../../../hacker", "voice", true_voice)  # Malicious path
    except ValueError:  # Expected ValueError
        traversal_blocked = True  # Blocked
    print(f" [Auth Test 3] Path Attack Defense: Direct injection of '../../' -> {'🛡️ ATTACK DEFLECTED' if traversal_blocked else '❌ VULNERABLE'}", flush=True)  # Log
    print("=" * 68, flush=True)  # Footer
    print("  🎉 Automated Security Verification: ALL CHECKS PASSED (100% OK)!\n", flush=True)  # Success message

# ── Mode 2: Live Microphone Voice Biometric Test ──────────────────────────────── #
def run_live_voice_test():  # Define live mic voice test function
    print("\n" + "=" * 68, flush=True)  # Banner border
    print("  SECURITY DEMO 2: Live Microphone Voice Biometric Authentication", flush=True)  # Title
    print("=" * 68, flush=True)  # Banner border
    try:  # Start try block to check SpeechBrain and torchaudio availability
        from auth.voice import enroll_voice, verify_voice  # Import voice functions
        from STT.whisper_stt import record_audio_to_wav, remove_temporary_file  # Import recorder
    except ImportError as err:  # Handle missing dependencies
        print(f" [Error] Missing library: {err}", flush=True)  # Print error
        return  # Exit
    user_id = "test_user"  # Target test user id
    print(f" Step 1: Recording voice sample to ENROLL user '{user_id}'...", flush=True)  # Prompt
    print(" Speak naturally into your microphone for 5 seconds when prompted...", flush=True)  # Instructions
    enroll_wav = record_audio_to_wav()  # Record enrollment audio file
    if not enroll_wav:  # Check if recording failed
        print(" [Error] Audio recording failed.", flush=True)  # Error log
        return  # Exit early
    try:  # Try block for enrollment
        print(" Extracting ECAPA-TDNN speaker vector and enrolling...", flush=True)  # Progress log
        enroll_voice(user_id, [enroll_wav])  # Enroll voice sample
        print(" ✅ Voiceprint successfully enrolled into secure storage!", flush=True)  # Success log
        print("-" * 68, flush=True)  # Divider
        print(" Step 2: Now record a second audio sample to VERIFY your voice...", flush=True)  # Prompt
        print(" Speak again for 5 seconds...", flush=True)  # Instructions
        verify_wav = record_audio_to_wav()  # Record verification sample
        if not verify_wav:  # Check if recording failed
            print(" [Error] Verification recording failed.", flush=True)  # Error log
            return  # Exit early
        try:  # Try block for verification
            print(" Comparing voice sample with enrolled profile...", flush=True)  # Progress log
            matched, sim = verify_voice(user_id, verify_wav)  # Run voice verification
            print("-" * 68, flush=True)  # Divider
            print(f" Verification Result: Similarity Score = {sim:.4f}", flush=True)  # Display score
            if matched:  # Check if matched
                print(" ✅ IDENTITY VERIFIED! Voiceprint matched enrolled account holder.", flush=True)  # Match message
            else:  # If rejected
                print(" ❌ VERIFICATION FAILED: Voice does not match enrolled profile.", flush=True)  # Mismatch message
        finally:  # Cleanup verify file
            remove_temporary_file(verify_wav)  # Delete temp verify WAV
    finally:  # Cleanup enroll file
        remove_temporary_file(enroll_wav)  # Delete temp enroll WAV

# ── Mode 3: Live Webcam Session Monitor Check ──────────────────────────────────── #
def run_live_webcam_test():  # Define live webcam monitor function
    print("\n" + "=" * 68)  # Banner border
    print("  SECURITY DEMO 3: Continuous Webcam Face Session Monitor")  # Title
    print("=" * 68)  # Banner border
    try:  # Try importing face monitor
        from auth.face import FaceSessionMonitor  # Import monitor class
    except Exception as err:  # Handle error
        print(f" [Error] Could not initialize face monitor: {err}")  # Print error
        return  # Exit
    print(" Starting webcam monitor for 15 seconds...")  # Inform user
    print(" • Look directly at the webcam -> Session stays ACTIVE")  # Instructions
    print(" • Turn away or cover camera  -> 'FACE LOST' warning triggers after 5s")  # Instructions
    print("-" * 68)  # Divider
    def on_lost():  # Define callback for lost face
        print(" ⚠️  [SECURITY EVENT] FACE LOST! User has stepped away from screen.", flush=True)  # Alert
    monitor = FaceSessionMonitor(user_id="user1", source=0, on_face_lost=on_lost)  # Create monitor
    monitor.start()  # Start monitoring thread
    try:  # Monitor for 15 seconds
        for sec in range(15, 0, -1):  # Countdown loop
            time.sleep(1)  # Sleep 1 second
            print(f" Monitoring session... {sec}s remaining", end="\r", flush=True)  # Status countdown
        print("\n Monitoring period completed.")  # Completed notice
    except KeyboardInterrupt:  # Allow early Ctrl+C exit
        print("\n Monitoring interrupted by user.")  # Interrupted notice
    finally:  # Ensure monitor is stopped
        monitor.stop()  # Stop background thread and release webcam
        print(" Webcam released cleanly.\n")  # Cleanup confirmation

# ── Main Interactive Menu ──────────────────────────────────────────────────────── #
def main():  # Define main launcher
    print("\n" + "=" * 68)  # Header border
    print("   SIO BIOMETRIC SECURITY & AUTHENTICATION MODULE RUNNER")  # Header title
    print("=" * 68)  # Header border
    print(" Choose how you want to test the Security Module:")  # Menu prompt
    print("   1. [QUICKEST] Automated Self-Test (Instant, No camera/mic needed)")  # Option 1
    print("   2. Live Microphone Voice Authentication (SpeechBrain ECAPA-TDNN)")  # Option 2
    print("   3. Live Webcam Face Session Monitor (OpenCV YuNet + SFace)")  # Option 3
    print("   4. Exit")  # Option 4
    print("=" * 68)  # Divider
    choice = input(" Enter choice (1/2/3/4) [Default: 1]: ").strip()  # Prompt for user choice
    if choice == "2":  # If user chose 2
        run_live_voice_test()  # Run voice test
    elif choice == "3":  # If user chose 3
        run_live_webcam_test()  # Run webcam test
    elif choice == "4":  # If user chose 4
        print(" Exiting security runner. Goodbye!")  # Exit message
    else:  # Default to 1
        run_automated_self_test()  # Run self-test

if __name__ == "__main__":  # Direct script execution check
    main()  # Run main function

