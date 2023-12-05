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
import matplotlib.pyplot as plt


BATCH_SIZE = 512
EPOCHS = 200
traing_location = 'C:\\Users\\nnamd\\Documents\\Homework\\LBC Proejct\\emotion_network\\emotional_audio_dataset\\Training'
SAMPLE_RATE = 16000
NUM_SAMPLES= 48000
loss_totals = []
highest_lossses = []
accuracy_totals = []

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    correct = 0
    softmax = nn.Softmax(dim=1)
    highest_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        #Calculate loss
        logits = model.forward(inputs)
        #print(predictions, targets)
        loss = loss_fn(logits, targets)
        if highest_loss < loss.item():
            highest_loss = loss.item()
        loss_totals.append(loss.item())
        # Backpropogate loss and update weights
        print("Loss = {}".format(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = softmax(logits)
        output = torch.argmax(output,dim=1)
        correct += torch.sum(output==targets).item()
    accuracy = 100 * correct / 1152
    print("Accuracy = {}".format(accuracy))
    accuracy_totals.append(accuracy)
    highest_lossses.append(highest_loss)

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("----------------------------------------------")
    print("Training is done")
    




if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)
    
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft=400,
        n_mels = 64
    )
    
    emo = EmotionSpeechDataset(traing_location, spectrogram, NUM_SAMPLES, device=device)
    train_data_loader = DataLoader(emo, batch_size=BATCH_SIZE)

    #build model
    cnn = CNNetwork().to(device)

    #instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    #train model
    train(cnn, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    #store model
    torch.save(cnn.state_dict(), "new_2.pth")
    print("Model trained and stored at new_2.pth")
    t = list(range(0,EPOCHS))
    plt.plot(t, accuracy_totals, color='b')
    plt.show()

    plt.plot(t, highest_lossses, color='r')
    plt.show()