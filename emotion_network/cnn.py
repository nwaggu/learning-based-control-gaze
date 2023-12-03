#Ekman emotions: Happiness, Suprise, Sadness, Anger, Disgust, Fear
#just CNN or CNN + BLSTM + Attention 
#Dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
#Ref paper: https://arxiv.org/ftp/arxiv/papers/2204/2204.13601.pdf 
#More github ref: https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch/blob/master/notebooks/parallel_cnn_attention_lstm.ipynb
#Mel Spectrogram extraction: https://www.youtube.com/watch?v=TdnVE5m3o_0
#Youtube tutorial set on CNN for audio: https://www.youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm

from torch import nn
import torch
from torchsummary import summary
class CNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(7,7),
                stride=(2,2),
                padding=(2,2)
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(5,5),
                stride=(2,2),
                #padding=(2,2)
            ),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3,3),
                stride=(2,2),
                #padding=(2,2)
            ),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                stride=(2,2),
               # padding=(2,2)
            ),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                stride=(2,2),
               # padding=(2,2)
            ),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3,3),
                stride=(2,2),
                #padding=(2,2)
            ),
            nn.ReLU(),
        )
        self.con7 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3,3),
                stride=(2,2),
                #padding=(2,2)
            ),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(2304, 512)
        self.linear_2 = nn.Linear(512, 8)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        print(input_data.shape)
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        final = self.linear_2(logits)
        predictions = self.softmax(logits)
        return predictions





if __name__ =="__main__":
    cnn = CNNetwork()
    if torch.cuda.is_available():
        cnn.cuda()
    summary(cnn, (1,128,128))