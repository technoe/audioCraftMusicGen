#!/usr/bin/env python3
import os
from random_word import RandomWords
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import warnings

def generate_unique_filename(path):
    r = RandomWords()
    while True:
        # Generate a random word
        random_word = r.get_random_word()
        # Create a potential filename by appending the desired extension
        new_filename = f"{random_word}.wav"
        # Check if a file with this name already exists in the specified path
        if not os.path.exists(os.path.join(path, new_filename)):
            return random_word

def create(model_name,duration,prompt):
    print("Preparing model")
    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=duration)  # generate 160 seconds.

    print("Generating music")
    wav = model.generate(prompt,  progress=True)

    print("Writing file")
    file_prefix = generate_unique_filename(os.getcwd())
    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'music/{file_prefix}',
                    one_wav.cuda(), 
                    model.sample_rate, 
                    strategy="loudness",
                    loudness_compressor=True)
    
    return file_prefix

if __name__ == "__main__":
    warnings.simplefilter('ignore') 
    for i in range(0,3):
        print(create(
            model_name="facebook/musicgen-medium",
            duration=60,
            prompt=["halloween, metal, synth-wave, 80s, 180bpm"]))
