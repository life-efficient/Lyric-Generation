# %%
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import random
import torch.nn.functional as F
from tqdm import tqdm


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


class LyricDataset():
    def __init__(self, chunk_size=100):
        self.chunk_size = chunk_size
        with open('lyrics.txt', 'r') as file:
            txt = file.read()
        txt = txt.lower()

        self.tokeniser = Tokeniser(txt)

        self.X = torch.tensor(self.tokeniser.encode(txt))
        self.Y = torch.tensor(np.roll(self.X, -1, axis=0))

        self.vocab_size = len(set(txt))

    def __len__(self):
        return len(self.X) // self.chunk_size

    def __iter__(self):
        for idx in range(len(self)):
            k = np.random.randint(0, len(self.X)-chunk_size)
            slc = slice(k, k+self.chunk_size)
            yield self.X[slc], self.Y[slc]


class RNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super().__init__()
        # store input parameters in the object so we can use them later on
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # required functions for model
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.rnn = torch.nn.RNN(hidden_size, hidden_size,
                                n_layers, batch_first=True)  # TODO remove batch first
        self.decoder = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # encode our input into a vector embedding
        x = self.embedding(x.unsqueeze(0)).unsqueeze(0)
        # calculate the output from our rnn based on our input and previous hidden state
        output, self.hidden = self.rnn(x.view(1, 1, -1), self.hidden)
        # calculate our output based on output of rnn
        output = self.decoder(output.view(1, -1))

        return output

    def init_hidden(self):
        # TODO remove batch dim
        self.hidden = torch.zeros(self.n_layers, 1, self.hidden_size)

    def generate(self):
        self.init_hidden()
        current_token_id = torch.tensor(
            random.randint(0, self.vocab_size - 1)).unsqueeze(0)
        generated = []
        for idx in range(1000):
            predicted = self.forward(current_token_id)
            current_token_id = torch.argmax(predicted)
            generated.append(int(current_token_id))
        return generated


class RNNFullSeq(RNN):
    def forward(self, X):
        self.init_hidden(X.shape[0])
        embedding = self.embedding(X)
        outputs, hidden = self.rnn(embedding, self.hidden)
        # print(hidden.shape)
        # print(outputs.shape)
        predictions = self.decoder(outputs)
        # print("final output shape:", predictions.shape)
        return predictions

    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)

# def train_one_sequence(model, seq):

# def train_full_seq(model, epochs=1):


def train(model, dataset, epochs=1):
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # choose optimizer
    n_steps = 0
    for epoch in range(epochs):
        epoch_loss = 0  # stored the loss per epoch

        for seq_inputs, seq_targets in tqdm(dataset):

            # n_chunks = len(X) // chunk_size
            # for chunk_idx in range(n_chunks):
            loss = 0
            # sequence = random_chunk(chunk_size)

            if type(model) is RNN:
                model.init_hidden()
                sequence = zip(seq_inputs, seq_targets)

                # sequentially input each character in our sequence and calculate loss
                for current_token_id, next_token_id in sequence:
                    logits = model(current_token_id)
                    target = next_token_id.unsqueeze(0)
                    loss += F.cross_entropy(logits, target)
            elif type(model) is RNNFullSeq:
                # add batch dim TODO remove once using dataloader
                seq_inputs = seq_inputs.unsqueeze(0)
                # print(seq_inputs.shape)
                predictions = model(seq_inputs)
                # predictions = torch.argmax(predictions, dim=2)
                seq_targets = seq_targets.unsqueeze(0)

                # this part is important
                # cross entropy thinks that the dimension after the batch should either be a class idx or a distribution
                # in our case it's a timestep
                # so we need to treat each (batch, timestep) pair as a different batch
                # e.g. B=4, T=100 -> B'=B*T=400
                # (BxT, vocab_size)
                predictions = predictions.view(-1, predictions.shape[-1])
                seq_targets = seq_targets.view(-1)  # BxT targets all in a line
                loss = F.cross_entropy(predictions, seq_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/Train", loss.item(), n_steps)
            n_steps += 1

            epoch_loss += loss  # add the loss of this sequence to the loss of this epoch

        epoch_loss /= len(dataset)  # avg loss per chunk

        print('Epoch ', epoch, ' Avg loss/chunk: ', epoch_loss.item())
        generated_token_ids = model.generate()
        writer.add_text("Generated Text", dataset.tokeniser.decode(
            generated_token_ids)[:300], epoch)

        print("breaking for testing purposes")
        break  # for testing purposes


if __name__ == "__main__":

    # HYPER-PARAMS
    lr = 0.005
    epochs = 500
    chunk_size = 100  # the length of the sequences which we will optimize over

    hidden_size = 50
    n_layers = 2

    dataset = LyricDataset(chunk_size=chunk_size)
    n_tokens = len(dataset.tokeniser.id_to_token)
    # instantiate our model from the class defined earlier
    myrnn = RNNFullSeq(n_tokens, hidden_size, n_layers)
    train(myrnn, dataset, epochs)
    myrnn = RNN(n_tokens, hidden_size, n_layers)
    train(myrnn, dataset, epochs)

# %%
