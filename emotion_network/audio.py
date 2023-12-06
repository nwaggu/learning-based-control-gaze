import sounddevice as sd
from scipy.io.wavfile import write
from cnn import CNNetwork
import torch
import torchaudio
from custom_dataset import EmotionSpeechDataset
from torch.utils.data import DataLoader
from test_cnn import predict
import os 

fs = 48000 # Sample rate
seconds = 3  # Duration of recording
SAMPLE_RATE = 16000
NUM_SAMPLES = 48000


recording_location = 'N:\\Projects\\learning-based-control-gaze\\emotion_network\\recording'
sd.default.channels = 1
print("Starting Recording")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
class_mapping = [ "happy", "sad", "angry","surprised"]
sd.wait()  # Wait until recording is finished

write('recording/00-00-00-00-00-00-00.wav', fs, myrecording)  # Save as WAV file 
print("Ending Recording")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)


cnn_net = CNNetwork()
state_dict = torch.load("new_2.pth", map_location=torch.device('cpu'))
cnn_net.load_state_dict(state_dict)
cnn_net.eval()

spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft=400,
    n_mels = 64
)

validation_data = EmotionSpeechDataset(recording_location, spectrogram,NUM_SAMPLES, device=device)
data_loader = DataLoader(validation_data)
for step, (data, label) in enumerate(data_loader):
    predicted, expected = predict(cnn_net, data, label, class_mapping)
    print(f"Expected: {expected}")

os.remove('recording/00-00-00-00-00-00-00.wav')



# cnn_net
# data = torch.randn(1, 3, 24, 24) # Load your data here, this is just dummy data
# output = model(data)
# prediction = torch.argmax(output)





