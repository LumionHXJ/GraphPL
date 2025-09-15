# code from MUSE
import os
from typing import List
import pickle
import numpy as np
import torch

from utils.vocab import Vocabulary


def to_index(sequence: List[str], vocab, prefix="", suffix=""):
    """ convert code to index (each timestamp contains one token) """
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    sequence = torch.tensor(sequence)
    return sequence


def to_vector(sequence: List[List[str]], vocab, prefix="", suffix=""):
    """ convert code to multihot vector (each timestamp contains many tokens) """
    if prefix:
        sequence = [[prefix]] + sequence
    if suffix:
        sequence = sequence + [[suffix]]
    multihot_vector = torch.zeros(len(sequence), len(vocab))
    for i, tokens in enumerate(sequence):
        for token in tokens:
            multihot_vector[i, vocab(token)] = 1
    return multihot_vector


def read(file, dtype='float'):
    with open(file) as file:
        header = file.readline().split(' ')
        count = int(header[0])
        dim = int(header[1])
        matrix = np.empty((count, dim), dtype=dtype)
        for i in range(count):
            matrix[i] = np.fromstring(file.readline(), sep=' ', dtype=dtype)
    return matrix


class eICUTokenizer:
    def __init__(self):
        self.age_vocabs, self.age_vocabs_size = self._load_age_vocabs()
        self.gender_vocabs, self.gender_vocabs_size = self._load_gender_vocabs()
        self.ethnicity_vocabs, self.ethnicity_vocabs_size = self._load_ethnicity_vocabs()

    def _load_age_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in range(18, 90):
            word = word // 10 * 10
            vocabs.add_word(str(word))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_gender_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in ["Female", "Male", "Other", "Unknown", ""]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_ethnicity_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in ["African American", "Asian", "Caucasian", "Hispanic", "Native American", "Other/Unknown", ""]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def __call__(
            self,
            age: str,
            gender: str,
            ethnicity: str
    ):
        age = str(int(age) // 10 * 10)
        age = torch.tensor(self.age_vocabs(age))
        gender = torch.tensor(self.gender_vocabs(gender))
        ethnicity = torch.tensor(self.ethnicity_vocabs(ethnicity))
        return age, gender, ethnicity
