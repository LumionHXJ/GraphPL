# Code From MUSE
class Vocabulary(object):

    def __init__(self, init_words=None):
        self.init_words = init_words
        if init_words is None:
            init_words = ["<pad>", "<cls>", "<eos>", "<sep>", "<unk>"]
        self.word2idx = {word: idx for idx, word in enumerate(init_words)}
        self.idx2word = {idx: word for idx, word in enumerate(init_words)}
        assert len(self.word2idx) == len(self.idx2word)
        self.idx = len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            try:
                return self.word2idx["<unk>"]
            except KeyError:
                raise KeyError(f"word {word} not in vocab and <unk> not in vocab")
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
