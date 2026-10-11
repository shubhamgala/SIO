# SIO Voice-First Banking Assistant - End-to-End Voice Pipeline                   # File title comment
# Connects Module 1 (STT) → Module 2 (NLP) → Module 3 (Dialogue Manager).      # Module description comment
# Flow: Microphone → Whisper STT → Intent/Entity NLP → Dialogue → Confirm.     # Pipeline flow comment

import os  # Import os module for file path and system operations
import sys  # Import sys module for path resolution and system configurations

# Reconfigure standard output streams to UTF-8 to support Hindi Devanagari on Windows # Description comment
if hasattr(sys.stdout, 'reconfigure'):  # Check if stdout supports reconfiguring stream encoding
    sys.stdout.reconfigure(encoding='utf-8')  # Set stdout encoding to UTF-8 for displaying Hindi text properly
if hasattr(sys.stderr, 'reconfigure'):  # Check if stderr supports reconfiguring stream encoding
    sys.stderr.reconfigure(encoding='utf-8')  # Set stderr encoding to UTF-8 for displaying error messages properly
# Add project root directory to sys.path so modules can import cleanly           # Description comment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current folder to sys.path
# ------------------------------------------------------------------------------ # Section separator comment
# Import required functions from Module 1 (STT), Module 2 (NLP), Module 3 (DM) # Description comment
# ------------------------------------------------------------------------------ # Section separator comment
from STT.whisper_stt import load_whisper_model  # Import Whisper model loader function from STT module
from STT.whisper_stt import record_audio_to_wav  # Import audio recording function from STT module
from STT.whisper_stt import transcribe_audio  # Import speech transcription function from STT module
from STT.whisper_stt import remove_temporary_file  # Import temporary audio cleanup function from STT module
from NLP.intent_recognition import process_query  # Import NLP intent and entity processing function
from Backend.dialogue_manager import run_dialogue  # Import Module 3 dialogue manager function
# ------------------------------------------------------------------------------ # Section separator comment
# Master function to run the full voice-triggered banking assistant pipeline     # Description comment
# ------------------------------------------------------------------------------ # Section separator comment
def run_voice_pipeline():  # Define function to run end-to-end voice processing
    print("=" * 70)  # Print decorative top banner line
    print("  SIO Voice Banking Assistant - Full Pipeline (STT → NLP → Dialogue)  ")  # Print title banner
    print("=" * 70)  # Print decorative divider line
    model = load_whisper_model()  # Initialize and load the Whisper STT model into memory
    if model is None:  # Check if Whisper model initialization failed
        print("[SIO Pipeline] Error: Whisper speech recognition model could not be loaded.")  # Error log
        return  # Terminate pipeline early if model is unavailable
    temp_audio_file = record_audio_to_wav()  # Record 5 seconds of live voice audio from user microphone
    if not temp_audio_file:  # Check if microphone recording failed or was aborted
        print("[SIO Pipeline] Error: Audio recording failed. Check microphone connection.")  # Error log
        return  # Terminate pipeline early if recording failed
    try:  # Start try block to ensure temporary audio file is always cleaned up
        transcribed_text, detected_lang = transcribe_audio(model, temp_audio_file)  # Transcribe speech
        if not transcribed_text:  # Check if no speech was detected in the audio recording
            print("[SIO Pipeline] No speech detected from microphone. Please try again.")  # Notify user
            return  # Terminate pipeline early if transcription returned empty text
        print("=" * 70)  # Print section divider line
        print(f" [Step 1: STT] Transcribed Speech : \"{transcribed_text}\"")  # Display transcribed text
        print(f" [Step 1: STT] Detected Language  : {detected_lang}")  # Display detected language code
        print("=" * 70)  # Print section divider line
        nlp_result = process_query(transcribed_text)  # Send transcribed text into NLP Module 2
        print(f" [Step 2: NLP] Classified Intent : {nlp_result['intent']} (Confidence: {nlp_result['confidence']})")  # Intent
        print(f" [Step 2: NLP] Extracted Entities: {nlp_result['entities']}")  # Display extracted entities
        print(f" [Step 2: NLP] Engine Used       : {nlp_result['engine']}")  # Display NLP engine used
        print("=" * 70)  # Print section divider line
        print(" [Step 3: DM]  Starting Dialogue Manager...")  # Log start of Module 3
        dialogue_result = run_dialogue(nlp_result, model)  # Run Module 3 dialogue with NLP result
        print("=" * 70)  # Print section divider line
        print(f" [Step 3: DM]  Dialogue Status  : {dialogue_result['status'].upper()}")  # Final status
        print(f" [Step 3: DM]  Final Intent     : {dialogue_result['intent']}")  # Final intent after DM
        print(f" [Step 3: DM]  Final Entities   : {dialogue_result['entities']}")  # Final entities after DM
        print(f" [Step 3: DM]  Message          : {dialogue_result['message']}")  # Final message
        print("=" * 70)  # Print footer separator line
        if dialogue_result['status'] == 'confirmed':  # Check if user confirmed the transaction
            print("[SIO Pipeline] ✅ Transaction confirmed! Ready for Module 4 (Banking Logic).")  # Next step
        elif dialogue_result['status'] == 'cancelled':  # Check if user cancelled the transaction
            print("[SIO Pipeline] ❌ Transaction cancelled by user.")  # Log cancellation
        else:  # Handle any error status from the dialogue manager
            print(f"[SIO Pipeline] ⚠️  Dialogue ended with status: {dialogue_result['status']}")  # Log status
    finally:  # Ensure temporary audio recording file is deleted from disk in all cases
        remove_temporary_file(temp_audio_file)  # Clean up temporary WAV audio file from disk
# ------------------------------------------------------------------------------ # Section separator comment
# Script entry point to run voice assistant when main.py is executed directly    # Description comment
# ------------------------------------------------------------------------------ # Section separator comment
if __name__ == "__main__":  # Check if script is executed directly as the main script
    run_voice_pipeline()  # Launch the full three-module live voice pipeline
# ============================================================================== # End of main.py script
