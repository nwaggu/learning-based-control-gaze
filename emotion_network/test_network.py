import torch 
from ffnn_Tutorial import FeedForwardNet, download_mnist_datasets

#What you want the dataset to guess
#In our case it would be emotions probably!

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6"
    "7",
    "8",
    "9"
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model.forward(input)
        #Output of above is a 2D Tensor object
        #Dim 1 = # of samples, Dim 2 = # of classes
        predicted_index = predictions[0].argmax()
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        return predicted, expected




if __name__ == "__main__":
    #Load model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    #load validation data!
    _, validation_data = download_mnist_datasets()

    #get a sample of data??
    input, target = validation_data[0][0], validation_data[0][1]

    
    #make an inference?
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)
    print(f"Predicted: {predicted}, Expected: {expected}")

