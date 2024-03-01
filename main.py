# File: main.py
# Author: Alec Grace
# Created: 20 Feb 2024
# Purpose:
#   Driver class for training and testing a simple FeedForward network to perform part-of-speech tagging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import embeddings
import feed_nn
from my_dataset import MyDataset

# side note: paperswithcode tend to have datasets with research papers
# process:
#     word -> embedding -> linear feed forward -> output layer (soft max) -> cross entropy loss


def check_accuracy(loader, model, device):
    """
    Measures accuracy of a model given a dataset
    :param loader: DataLoader object
    :param model: Neural network
    :param device: "cuda" or "cpu"
    :return: Prints accuracy data to console, sets model back to train, and returns to program
    """
    correct = 0
    samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            correct += (predictions == y).sum()
            samples += predictions.size(0)
    print(f"{correct} / {samples} with accuracy {float(correct) / float(samples) * 100:.2f}%")
    model.train()
    return


def train(tuples, batch_size, net, device, optimizer, criterion, epochs):
    """
    Trains the given model on the given dataset with the given parameters
    :param tuples: List of (embedded vector, pos tag as index from mapping)
    :param batch_size: Int size of batches
    :param net: Model to train
    :param device: "cuda" or "cpu"
    :param optimizer: Optimizer of choice
    :param criterion: Loss function
    :param epochs: Epochs
    :return:
    """
    for j in range(epochs):
        for batch in DataLoader(MyDataset(tuples), batch_size=batch_size, shuffle=True):
            word, label = batch
            word = word.to(device=device)
            label = label.to(device=device)

            # feed forward
            scores = net(word)
            loss = criterion(scores, label)

            # back prop
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()
        return


def main():
    # store each transcript as numpy array of word vectors
    # planning on generalizing directories and dictionaries with config file eventually
    # ww_scripts = {}
    # for script in os.listdir("ww_text/"):
    #     ww_scripts[script] = embedder.get_tensor_embeddings("ww_text/" + script, True)
    # dn_scripts = {}
    # for script in os.listdir("dn_text/"):
    #     dn_scripts[script] = embedder.get_tensor_embeddings("dn_text/" + script, True)

    train_data = "words_pos.csv"
    with open(train_data, 'r') as data:
        datalines = data.readlines()
    data.close()
    train_tuples = []
    pos_mapping = []
    for line in datalines:
        pos = line.split(',')[2].rstrip('\n')
        pos_mapping.append(pos)
    pos_mapping = sorted(set(pos_mapping))

    # Instantiate an embedder
    embedder = embeddings.Embeddings()

    # Load data
    for line in datalines:
        word = line.split(',')[1]
        pos = line.split(',')[2].rstrip('\n')
        try:
            train_tuples.append((embedder.get_embedding(word), pos_mapping.index(pos)))
        except KeyError:
            continue
    # Split list into cross validation sets here
    folds = 100
    split_data = []
    for i in range(len(train_tuples) // folds):
        split_data.append(train_tuples[i:i + len(train_tuples) // 100])

    # Initialize network & parameters
    epochs = 500000
    learning_rate = 0.001
    batch_size = 2000
    criterion = nn.CrossEntropyLoss()
    net = feed_nn.FeedNet(300, 100, 25)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train network
    for i in range(len(split_data)):
        validate = split_data[i] # check accuracy with this
        training = []
        for item in split_data[:i]:
            training.extend(item)
        for item in split_data[i + 1:]:
            training.extend(item)
        train(training, batch_size, net, device, optimizer, criterion, epochs)
        # check accuracy
        check_accuracy(DataLoader(MyDataset(validate)), net, device)


if __name__ == "__main__":
    main()
