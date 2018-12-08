from nltk.stem import PorterStemmer
from torch.utils.data import Dataset, DataLoader


def estimate_subword(word, vocab, n_values):
    min_word, min_freq = None, len(vocab)
    for n in n_values:
        for i in range(len(word) - n + 1):
            subword = word[i:i+n]
            if subword in vocab:
                if vocab[subword] < min_freq:
                    min_word, min_freq = subword, vocab[subword]
    return min_word



class WordSimDataset(Dataset):
    def __init__(self, filename, vocab, vocab_idx, n_values=range(3, 7)):
        self.ps = PorterStemmer()
        self.data = []
        num = -1
        with open(filename, "r") as f:
            num = -1
            for line in f:
                num += 1
                if num == 0:
                    continue
                all_line = line.replace('\n', '').split(',')
                first, second = all_line[0], all_line[1]
                scores = all_line[3:]
                if first not in vocab:
                    temp = self.ps.stem(first)
                    if temp not in vocab:
                        temp = estimate_subword(first, vocab, n_values)
                        if temp is None:
                            continue
                    first = temp
                if second not in vocab:
                    temp = self.ps.stem(second)
                    if temp not in vocab:
                        temp = estimate_subword(second, vocab, n_values)
                        if temp is None:
                            continue
                    second = temp
                i1, i2 = vocab_idx[first] , vocab_idx[second]
                assert i1 < len(vocab), i1
                assert i2 < len(vocab), i2
                self.data.extend([((i1, i2), float(score))
                                  for score in scores])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
