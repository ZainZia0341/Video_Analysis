import os
from pydub import AudioSegment
import wave
import numpy as np
import deepspeech

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    print("checking audio load ", audio)
    audio.export(wav_file_path, format="wav")
    print(f"Converted {mp3_file_path} to {wav_file_path}")

def transcribe_audio(wav_file_path, model, scorer):
    print(f"Starting transcription for {wav_file_path}...")
    wf = wave.open(wav_file_path, "rb")
    fs = wf.getframerate()
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    
    print("Enabling scorer...")
    model.enableExternalScorer(scorer)
    print("Running model.stt...")
    text = model.stt(audio)
    
    print(f"Transcription for {wav_file_path}:")
    print(text)
    return text

def process_audio_files(input_dir, output_dir, model_path, scorer_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading DeepSpeech model from {model_path}...")
    model = deepspeech.Model(model_path)
    print(f"Loading DeepSpeech scorer from {scorer_path}...")
    model.enableExternalScorer(scorer_path)
    
    for index, audio_file in enumerate(os.listdir(input_dir), start=1):
        if audio_file.endswith('.mp3'):
            mp3_file_path = os.path.join(input_dir, audio_file)
            wav_file_path = os.path.join(output_dir, f'audio{index}.wav')
            convert_mp3_to_wav(mp3_file_path, wav_file_path)
            transcription = transcribe_audio(wav_file_path, model, scorer_path)
            text_file_path = os.path.join(output_dir, f'textfile{index}.txt')
            with open(text_file_path, 'w') as f:
                f.write(transcription)
            print(f"Transcription saved to {text_file_path}")

# Directories
input_dir = 'extracted_audio'
output_dir = 'transcription_results'
model_path = 'audio_model/deepspeech-0.9.3-models.pbmm'  # Replace with your DeepSpeech model path
scorer_path = 'audio_model/deepspeech-0.9.3-models.scorer'  # Replace with your DeepSpeech scorer path

process_audio_files(input_dir, output_dir, model_path, scorer_path)
