# %%
from pprint import pprint
import torch
import torch.nn.functional as F
from dataset import LyricDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random


# %%
class RNN(torch.nn.Module):
    def __init__(self,
                 n_embeddings,
                 bidirectional=False,
                 embedding_dim=128,
                 hidden_size=64,
                 n_layers=2,
                 ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            n_embeddings,
            embedding_dim=embedding_dim
        )
        self.encoder = torch.nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=bidirectional,
            # nonlinearity='relu',
            batch_first=True
            # dropout=0.5
        )
        self.decoder = torch.nn.Linear(hidden_size, n_embeddings)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_embeddings = n_embeddings
        self.init_hidden()

    def init_hidden(self):
        # return torch.zeros(n_layers, 1, hidden_size) # batched inputs
        # non-batched inputs
        self.hidden = torch.zeros(self.n_layers, self.hidden_size)

    def forward(self, token_id):
        token_id = torch.tensor(token_id).unsqueeze(0)
        embedding = self.embedding(token_id)
        output, self.hidden = self.encoder(embedding, self.hidden)
        return self.decoder(output)

    def generate(self):
        current_token_id = random.randint(
            0, self.n_embeddings)  # set as CLS token
        self.init_hidden()
        predicted_sequence = [current_token_id]
        for idx in range(100):
            # TODO stop on EOS token
            logits = self.forward(current_token_id)
            current_token_id = torch.argmax(logits)
            print(current_token_id)
            predicted_sequence.append(current_token_id)
        return predicted_sequence


def train(model, dataset, epochs=10):
    writer = SummaryWriter()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.0001)
    # for epoch in epochs:
    batch_idx = 0
    for sequence in dataset:
        # print(sequence)
        # print(dataset.tokeniser.id_to_token(sequence[0]))
        model.hidden.detach()
        model.init_hidden()
        loss = 0
        for idx in range(len(sequence)-1):
            current_token_id = sequence[idx]
            next_token_id = torch.tensor(
                sequence[idx+1]).unsqueeze(0)  # teacher forcing
            # print(next_token_id)
            # print(token_id)
            # print(dataset.tokeniser.id_to_token(token_id))

        # for epoch in range(epochs):
        # for batch in loader:
            predictions = model(current_token_id)
            # print("predicted word:", dataset.tokeniser.id_to_token(
            #     torch.argmax(predictions)))
            # print(predictions)
            # print(predictions.shape)
            # print(next_token_id.shape)
            loss += F.cross_entropy(predictions, next_token_id)
        # print('END OF SEQUENCE')
        writer.add_scalar("Loss/Train", loss.item(), batch_idx)
        batch_idx += 1

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        # print("Batch:", batch_idx, "Loss:", loss.item())
        # print()

        if batch_idx % 50 == 0:
            generated_sequence = model.generate()
            print(dataset.tokeniser.decode(generated_sequence))


dataset = LyricDataset(vocab_size=10000)
n_embeddings = dataset.tokeniser.get_vocab_size()
# loader = DataLoader(dataset, shuffle=True, batch_size=4)
model = RNN(n_embeddings)

train(model, dataset)
# %%
pprint(dataset.tokeniser.id_to_token(0))
# %%
