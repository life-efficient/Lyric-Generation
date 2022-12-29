# %%
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import random


def create_vocab(rawtxt):
    letters = list(set(rawtxt))
    lettermap = dict(enumerate(letters))  # created the dictionary mapping

    return lettermap


class Tokeniser:
    def __init__(self, txt):

        unique_chars = set(txt)
        self.id_to_token = dict(enumerate(unique_chars))
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

    def encode(self, str):
        return [self.token_to_id[char] for char in str.strip().lower()]

    def decode(self, token_ids):
        return "".join([self.id_to_token[id] for id in token_ids])

# return a random batch for training


def random_chunk(chunk_size):
    k = np.random.randint(0, len(X)-chunk_size)
    return X[k:k+chunk_size], Y[k:k+chunk_size]


class RNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super().__init__()
        # store input parameters in the object so we can use them later on
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # required functions for model
        self.encoder = torch.nn.Embedding(vocab_size, hidden_size)
        self.rnn = torch.nn.RNN(hidden_size, hidden_size,
                                n_layers, batch_first=True)
        self.decoder = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # encode our input into a vector embedding
        x = self.encoder(x.view(1, -1))
        # calculate the output from our rnn based on our input and previous hidden state
        output, self.hidden = self.rnn(x.view(1, 1, -1), self.hidden)
        # calculate our output based on output of rnn
        output = self.decoder(output.view(1, -1))

        return output

    def init_hidden(self):
        self.hidden = torch.zeros(self.n_layers, 1, self.hidden_size)

    def generate(self):
        self.init_hidden()
        current_token_id = torch.tensor(
            random.randint(0, self.vocab_size)).unsqueeze(0)
        generated = []
        for idx in range(1000):
            predicted = self.forward(current_token_id)
            current_token_id = torch.argmax(predicted)
            generated.append(int(current_token_id))
        return generated

# def train_one_sequence(model, seq):


def train(model, epochs=1):
    writer = SummaryWriter()
    criterion = torch.nn.CrossEntropyLoss()  # define our loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # choose optimizer
    n_steps = 0
    for epoch in range(epochs):
        epoch_loss = 0  # stored the loss per epoch

        # given our chunk size, how many chunks do we need to optimizer over to have gone thorough our whole dataset
        n_chunks = len(X)//chunk_size
        for chunk_idx in range(n_chunks):
            model.init_hidden()
            loss = 0

            x, y = random_chunk(chunk_size)

            # sequentially input each character in our sequence and calculate loss
            for i in range(chunk_size):
                out = model.forward(x[i])
                target = y[i].unsqueeze(0)
                loss += criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/Train", loss.item(), n_steps)
            n_steps += 1

            epoch_loss += loss  # add the loss of this sequence to the loss of this epoch

        epoch_loss /= n_chunks  # avg loss per chunk

        print('Epoch ', epoch, ' Avg loss/chunk: ', epoch_loss.item())
        # print('Generated text: ', generated, '\n')
        generated_token_ids = model.generate()
        writer.add_text("Generated Text", tokeniser.decode(
            generated_token_ids)[:300], epoch)


if __name__ == "__main__":
    with open('lyrics.txt', 'r') as file:
        txt = file.read()
    txt = txt.lower()

    tokeniser = Tokeniser(txt)

    X = tokeniser.encode(txt)
    Y = np.roll(X, -1, axis=0)
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    n_tokens = len(set(txt))

    # HYPER-PARAMS
    lr = 0.005
    epochs = 500
    chunk_size = 100  # the length of the sequences which we will optimize over

    # instantiate our model from the class defined earlier
    myrnn = RNN(n_tokens, 50, 2)
    train(myrnn, epochs)

# %%
