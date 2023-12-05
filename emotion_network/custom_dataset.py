from torch.utils.data import Dataset
import os 
import torchaudio
import torch
import opensmile
from scipy.io import wavfile
import noisereduce as nr
from torch.utils.data import DataLoader
import random


# load data


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)


class EmotionSpeechDataset(Dataset):

    def __init__(self, audio_dir, transformation, number_samples, device,sample_rate=16000):
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.sample_rate = sample_rate
        self.number_of_samples = number_samples
        self.folder = os.listdir(self.audio_dir)
        random.shuffle(self.folder)


    def __len__(self):
        count = 0
        for root_dir, cur_dir, files in os.walk(self.audio_dir):
            count += len(files)
        return count

    def __getitem__(self, index):
        file = self.folder[index]
        audio_sample_path = self._get_audio_sample_path(file)
        label = self._get_audio_sample_label(file)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        #Extra processing if necessary
        #Confirm sample rate
        signal = self._resample(signal, sr)
        #Combine channels
        signal = self._mix_down(signal)
        #More samples than expected
        signal = self._cut(signal)
        #Less samples than expected
        signal = self._right_pad(signal)
        #Processing End
        #df = smile.process_file(audio_sample_path)
        #print(len(df.values[0]))
        #output = self.df_to_tensor(df)
        signal = self.transformation(signal)
        return signal, label 

    def _cut(self, signal):
        if signal.shape[1] > self.number_of_samples:
            signal = signal[:, :self.number_of_samples]
        return signal

    def _right_pad(self, signal):
        
        length_signal = signal.shape[1]
        if length_signal < self.number_of_samples:
            missing_samples = self.number_of_samples - length_signal
            last_dim_padding = (0, missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
        
    def _resample(self, signal, sr):
        if sr != self.sample_rate:
            resample = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            signal = resample(signal)
        return signal

    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    
    def _get_audio_sample_path(self, file):
        path = os.path.join(self.audio_dir, file)
        return path
    
    def _get_audio_sample_path_new(self, dir, file):
        path = os.path.join(dir, file)
        return path

    def _get_audio_sample_label(self, file):
        lab = int(file[6:8])
        if lab == 1:
            return 0
        elif lab == 3:
            return 0
        elif lab == 4:
            return 1
        elif lab == 5:
            return 2
        elif lab == 8:
            return 3
        else:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~impossible")
            return 0
    

    def df_to_tensor(self, df):
        return torch.from_numpy(df.values).float().to(self.device)





if __name__ == "__main__":
    SAMPLE_RATE=16000
    traing_location = 'C:\\Users\\nnamd\\Documents\\Homework\\LBC Proejct\\emotion_network\\emotional_audio_dataset\\Training'
    new_location = 'B:\\Temp\\New_Training'

    val_location = 'C:\\Users\\nnamd\\Documents\\Homework\\LBC Proejct\\emotion_network\\emotional_audio_dataset\\Training'
    new_val = 'B:\\Temp\\New_Validation'
    NUM_SAMPLES = 48000


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)


    #Instantiate Dataset
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft=400,
        n_mels = 64
    )
    
    emo = EmotionSpeechDataset(traing_location, spectrogram, NUM_SAMPLES, device=device)
    train_data_loader = DataLoader(emo, batch_size=1)

    for inputs, targets in train_data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        print(targets)
