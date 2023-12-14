import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

Fs, aud = wavfile.read('angry.wav')
plt.figure().set_figheight(2)
powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)

plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (Seconds)")
plt.show()