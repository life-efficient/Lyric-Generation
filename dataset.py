# %%
import random
from torch.utils.data import Dataset
import os
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from glob import glob
from datasets import load_dataset


class LyricDataset(Dataset):
    def __init__(self, root="data", seq_len=16, vocab_size=500):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.song_paths = glob(os.path.join(root, "*", "*"))
        self.tokeniser = self.get_tokeniser()

        self.tokenised_songs = []
        # n_examples = 0
        for song_path in self.song_paths:
            with open(song_path) as f:
                lyrics = f.read()
            tokenised_song = self.tokeniser.encode(lyrics)
            # print(tokenised_song.tokens)
            self.tokenised_songs.append(tokenised_song)  # tokenise
        random.shuffle(self.tokenised_songs)

    def get_tokeniser(self):
        tokeniser = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokeniser.normalizer = normalizers.Sequence(
            [NFD(), Lowercase(), StripAccents()])
        tokeniser.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=[
            "[UNK]", "[CLS]", "[SEP]", "[PAD]"])

        tokeniser.train(self.song_paths, trainer)
        return tokeniser

    def __iter__(self):
        while True:
            random_song_encoding = random.choice(self.tokenised_songs)
            # print(random_song_encoding.ids)
            random_song_tokens = random_song_encoding.ids
            start_token_idx = random.randint(
                0, len(random_song_tokens) - self.seq_len)
            random_lyric_sequence = random_song_tokens[start_token_idx:start_token_idx+self.seq_len]
            yield random_lyric_sequence

    def __len__(self):
        return


# class WikiTextDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         dataset = load_dataset("wikitext", 'wikitext-103-v1',
#                                split='test')

        # self.examples =


if __name__ == "__main__":
    dataset = LyricDataset()
    for example in dataset:
        print(example)
        scsd

# %%
