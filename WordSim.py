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



def WordSimDataset(Dataset):
    def __init__(self, filename, vocab, n_values=range(3, 7)):
        self.ps = PorterStemmer()
        self.data = []
        skip = 0
        with open(filename, "r") as f:
            for line in [line.strip().split(',') for line in f]:
                skip +=1 
                if skip == 1:
                    continue 

                first, second = line[0], line[1]
                scores = line[2:]
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
                i1, i2 = vocab[first] , vocab[second]
                self.data.extend([((i1, i2), float(score))
                                  for score in scores])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
