from torch.utils.data import Dataset
import os 
import torchaudio
import torch
import opensmile


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)


class EmotionSpeechDataset(Dataset):

    def __init__(self, audio_dir, transformation, sample_rate, number_samples, device):
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.sample_rate = sample_rate
        self.number_of_samples = number_samples


    def __len__(self):
        count = 0
        for root_dir, cur_dir, files in os.walk(self.audio_dir):
            count += len(files)
        return count

    def __getitem__(self, index):
        folder = os.listdir(self.audio_dir)
        file = folder[index]
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
            resample = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resample(signal)
        return signal

    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    
    def _get_audio_sample_path(self, file):
        path = os.path.join(self.audio_dir, file)
        return path

    def _get_audio_sample_label(self, file):
        return int(file[6:8])-1
    

    def df_to_tensor(self, df):
        return torch.from_numpy(df.values).float().to(self.device)





