import torch 
from cnn import CNNetwork
from custom_dataset import EmotionSpeechDataset
from torch.utils.data import DataLoader
import torchaudio
from torchsummary import summary

#What you want the dataset to guess
#In our case it would be emotions probably!

class_mapping = [
"neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        print(input.shape)
        predictions = model(input)
        #Output of above is a 2D Tensor object
        #Dim 1 = # of samples, Dim 2 = # of classes
        predicted_index = predictions[0].argmax()
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        print("complete")
        return predicted, expected




if __name__ == "__main__":
    #Load model
    cnn_net = CNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn_net.load_state_dict(state_dict)


    #audio = 'C:\\Users\\nnamd\\Documents\\Homework\\LBC Proejct\\emotion_network\\emotional_audio_dataset\\Training'
    validation_location = 'C:\\Users\\nnamd\\Documents\\Homework\\LBC Proejct\\emotion_network\\emotional_audio_dataset\\Validation'
    SAMPLE_RATE = 28300
    NUM_SAMPLES=85000


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    #ms = mel_spectrogram(signal)
    count= 0
    correct = 0
    validation_data = EmotionSpeechDataset(validation_location, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    train_data_loader = DataLoader(validation_data, batch_size=1)
    for step, (data, label) in enumerate(train_data_loader):
        print(data.shape)
        predicted, expected = predict(cnn_net, data, label, class_mapping)
        print(f"Predicted: {predicted}, Expected: {expected}")
        if predicted == expected:
            correct+=1
        count+=1
    print(correct/count)
    #make an inference?
    #pr
    #

