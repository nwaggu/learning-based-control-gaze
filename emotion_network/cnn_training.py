#Ekman emotions: Happiness, Suprise, Sadness, Anger, Disgust, Fear
#just CNN or CNN + BLSTM + Attention 
#Dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
#Ref paper: https://arxiv.org/ftp/arxiv/papers/2204/2204.13601.pdf 
#More github ref: https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch/blob/master/notebooks/parallel_cnn_attention_lstm.ipynb
#Mel Spectrogram extraction: https://www.youtube.com/watch?v=TdnVE5m3o_0
from torch import nn
import torch 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence
from custom_dataset import EmotionSpeechDataset
import torchaudio
from cnn import CNNetwork

BATCH_SIZE = 128
EPOCHS = 10
audio = 'C:\\Users\\nnamd\\Documents\\Homework\\LBC Proejct\\emotion_network\\emotional_audio_dataset'
SAMPLE_RATE = 28300
NUM_SAMPLES=85000

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        #Calculate loss
        predictions = model.forward(inputs)
        #print(predictions, targets)
        loss = loss_fn(predictions, targets)
        
        # Backpropogate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("----------------------------------------------")
    print("Training is done")
    pass




if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)


    #Instantiate Dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    emo = EmotionSpeechDataset(audio, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    #Create a Data Loader for the Train set
    train_data_loader = DataLoader(emo, batch_size=BATCH_SIZE)

    #build model
    cnn = CNNetwork().to(device)

    #instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters())

    #train model
    train(cnn, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    #store model
    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and stored at cnn.pth")