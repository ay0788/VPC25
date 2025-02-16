import os
import subprocess
import whisper
import numpy as np
import soundfile as sf

# Initialize Whisper model globally (to avoid reloading for each call)
whisper_model = whisper.load_model("base")

# Define the text-to-speech function (Piper)
def text_to_speech(text, output_file, model_path=r"C:\Users\sdour\Desktop\ndsc\national-data-science-competition\competitionn\piper\voices\Ten_US-lessac-medium.onnx"):
    piper_path = r"C:\Users\sdour\Desktop\ndsc\national-data-science-competition\competitionn\piper\piper.exe"
    if not os.path.exists(piper_path): 

        raise FileNotFoundError(f"Piper executable not found at: {piper_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    command = [
        piper_path, "--model", model_path, "--output_file", output_file
    ]
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=text)
    
    if process.returncode != 0:
        raise RuntimeError(f"Piper Error: {stderr.strip()}")

# Implement the anonymization function
def anonymize(input_audio_path):
    """
    Anonymizes speech by converting it to text and generating a new voice output.
    
    Parameters
    ----------
    input_audio_path : str
        Path to the input audio file (.wav).
    
    Returns
    -------
    audio : numpy.ndarray
        The anonymized speech as a NumPy array (float32).
    sr : int
        The sample rate of the processed audio.
    """
    # Step 1: Transcribe speech to text
    try:
        result = whisper_model.transcribe(input_audio_path)
        transcribed_text = result["text"]
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")
    
    # Step 2: Convert text to speech
    output_audio_path = r"C:\Users\sdour\Desktop\test\VPC25\evaluation_data\Enrollment\speaker1\1272-128104-0003.wav"
    text_to_speech(transcribed_text, output_audio_path)

    # Step 3: Load the generated audio file and return it
    audio, sr = sf.read(output_audio_path, dtype="float32")
    return audio, sr