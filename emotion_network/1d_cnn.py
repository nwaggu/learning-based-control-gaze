#Ekman emotions: Happiness, Suprise, Sadness, Anger, Disgust, Fear
#just CNN or CNN + BLSTM + Attention 
#Dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
#Ref paper: https://arxiv.org/ftp/arxiv/papers/2204/2204.13601.pdf 
#More github ref: https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch/blob/master/notebooks/parallel_cnn_attention_lstm.ipynb
#Mel Spectrogram extraction: https://www.youtube.com/watch?v=TdnVE5m3o_0
#Youtube tutorial set on CNN for audio: https://www.youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm

from torch import nn
from torchsummary import summary
class CNNetwork1D(nn.Module):

    def __init__(self):
        super().__init__()
        #4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(4032, 8)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        x = self.conv1(input_data)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ =="__main__":
    cnn = CNNetwork1D()
    summary(cnn, (1,988))