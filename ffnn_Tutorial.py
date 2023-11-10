#Ekman emotions: Happiness, Suprise, Sadness, Anger, Disgust, Fear
#just CNN or CNN + BLSTM + Attention 
#Dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
#Ref paper: https://arxiv.org/ftp/arxiv/papers/2204/2204.13601.pdf 
#More github ref: https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch/blob/master/notebooks/parallel_cnn_attention_lstm.ipynb
#Mel Spectrogram extraction: https://www.youtube.com/watch?v=TdnVE5m3o_0
from torch import nn
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#TUTORIAL

BATCH_SIZE = 128
EPOCHS = 10

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root = "data",
        download = True,
        train = True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root = "data",
        download = True,
        train = True,
        transform=ToTensor()
    )
    return train_data, validation_data




class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() 
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        #Calculate loss
        predictions = model(inputs)
        print(predictions, targets)
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

    train_data, validation_data = download_mnist_datasets()
    print("Data Downloaded :)")

    #Create a Data Loader for the Train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    #build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    feed_forward_net = FeedForwardNet().to(device)

    #instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters())

    #train model
    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    #store model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedforwardnet.pth")