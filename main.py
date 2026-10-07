# SIO Voice-First Banking Assistant - End-to-End Voice Pipeline                   # File title comment
# Connects Module 1 (Speech to Text) directly with Module 2 (Intent & Entities). # Module description comment
# Flow: Microphone Audio -> Whisper STT -> IndicBERT/NLP Intent Classifier.     # Pipeline flow comment

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
# Import required functions from Module 1 (STT) and Module 2 (NLP)               # Description comment
# ------------------------------------------------------------------------------ # Section separator comment
from STT.whisper_stt import load_whisper_model  # Import Whisper model loader function
from STT.whisper_stt import record_audio_to_wav  # Import audio recording function
from STT.whisper_stt import transcribe_audio  # Import speech transcription function
from STT.whisper_stt import remove_temporary_file  # Import temporary audio cleanup function
from NLP.intent_recognition import process_query  # Import NLP intent and entity processing function
# ------------------------------------------------------------------------------ # Section separator comment
# Master function to run the voice-triggered banking assistant pipeline          # Description comment
# ------------------------------------------------------------------------------ # Section separator comment
def run_voice_pipeline():  # Define function to run end-to-end voice processing
    print("=" * 70)  # Print decorative top banner line
    print("  SIO Voice Banking Assistant - Live Voice Triggered Pipeline  ")  # Print title banner
    print("=" * 70)  # Print decorative divider line
    model = load_whisper_model()  # Initialize and load the Whisper STT model
    if model is None:  # Check if model initialization failed
        print("[SIO Pipeline] Error: Whisper speech recognition model could not be loaded.")  # Error log
        return  # Terminate pipeline early
    temp_audio_file = record_audio_to_wav()  # Record 5 seconds of live voice audio from user microphone
    if not temp_audio_file:  # Check if microphone recording failed or was aborted
        print("[SIO Pipeline] Error: Audio recording failed. Check microphone.")  # Error log
        return  # Terminate pipeline early
    try:  # Start try block to ensure audio file cleanup
        transcribed_text, detected_lang = transcribe_audio(model, temp_audio_file)  # Transcribe user speech to text
        if not transcribed_text:  # Check if no speech was detected in the audio
            print("[SIO Pipeline] No speech detected from microphone. Please try again.")  # Notify user
            return  # Terminate pipeline early
        print("=" * 70)  # Print section divider line
        print(f" [Step 1: STT] Transcribed Speech : \"{transcribed_text}\"")  # Display transcribed voice text
        print(f" [Step 1: STT] Detected Language  : {detected_lang}")  # Display detected language code
        print("=" * 70)  # Print section divider line
        nlp_result = process_query(transcribed_text)  # Send live transcribed text into NLP Module 2
        print(f" [Step 2: NLP] Classified Intent : {nlp_result['intent']} (Confidence: {nlp_result['confidence']})")  # Intent
        print(f" [Step 2: NLP] Extracted Entities: {nlp_result['entities']}")  # Display extracted entities
        print(f" [Step 2: NLP] Engine Used       : {nlp_result['engine']}")  # Display NLP processing engine
        print("=" * 70)  # Print section divider line
        print("[SIO Pipeline] Ready for Module 3 (Banking Actions & Voice Response)!")  # Next step log
    finally:  # Ensure temporary audio recording file is deleted from disk
        remove_temporary_file(temp_audio_file)  # Clean up temporary audio file
# ------------------------------------------------------------------------------ # Section separator comment
# Script entry point to run voice assistant when main.py is executed directly    # Description comment
# ------------------------------------------------------------------------------ # Section separator comment
if __name__ == "__main__":  # Check if executed directly as main script
    run_voice_pipeline()  # Launch live voice pipeline
# ============================================================================== # End of main.py script
# ------------------------------------------------------------------------------ # End of file marker

