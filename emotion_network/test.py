
from scipy.io import wavfile
import noisereduce as nr
import os
import time

folder = os.listdir("C:\\Users\\nnamd\\Documents\\Homework\\LBC Proejct\\emotion_network\\emotional_audio_dataset\\Training")
i = 0
for file in folder:
    i+=1
    rate, data = wavfile.read(f"C:\\Users\\nnamd\\Documents\\Homework\\LBC Proejct\\emotion_network\\emotional_audio_dataset\\Training\\{file}")
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=16000)
    new_path = f"B:\\Temp\\New_Training\\{file}"
    wavfile.write(new_path, rate, reduced_noise)