import numpy as np
import torch
from collections import Counter
from torchtext.vocab import vocab

# Specify Data Folder


class CustomDataset(torch.utils.data.Dataset):
    """IMDB dataset."""

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        texts = self.X[idx]
        labels = self.y[idx]
        sample = (labels, texts)

        return sample


def get_vocab(dataset, min_freq=1):
    counter = Counter()
    for (l_, text) in dataset:
        counter.update(str(text).split())

    # creating vocab using the vocab object from trochtext
    my_vocab = vocab(counter, min_freq=min_freq)

    # insert '<unk>' token to represent any unknown word
    my_vocab.insert_token('<unk>', 0)

    # set the default index to zero
    # thus any unknown word will be represented b index 0 or token '<unk>'
    my_vocab.set_default_index(0)

    return my_vocab

# Creating a function that will be used to get the indices of words from vocab


def text_pipeline(x, vocab):
    """Converts text to a list of indices using a vocabulary dictionary"""
    return [vocab[token] for token in str(x).split()]

# Create Collate Function


def collate_batch(batch, vocab):
    labels, texts = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.float64)
    list_of_list_of_indices = [text_pipeline(text, vocab) for text in texts]

    offsets = [0] + [len(i) for i in list_of_list_of_indices]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    texts = torch.cat([torch.tensor(i, dtype=torch.int64)
                      for i in list_of_list_of_indices])
    return (texts, offsets), labels


def get_loaders(trainset, validset, batch_size_, collate_fn):

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_, shuffle=True,
                                               collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size_, shuffle=False,
                                               collate_fn=collate_fn)
    return train_loader, valid_loader


def get_test_loaders(testset, batch_size_, collate_fn):

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_,   shuffle=False,
                                              collate_fn=collate_fn)
    return test_loader
