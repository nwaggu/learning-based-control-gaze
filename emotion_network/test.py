
from scipy.io import wavfile
import noisereduce as nr
import os
import time

rate, data = wavfile.read("output.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
new_path = "output.wav"
wavfile.write(new_path, rate, reduced_noise)