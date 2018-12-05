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

class RareWordDataset(Dataset):
    def __init__(self, filename, vocab, n_values=range(3,7)):
        self.ps = PorterStemmer()
        self.data = []
        with open(filename, "r") as f:
            for line in [line.strip().lower().split() for line in f]:
                word_pair = (line[0], line[1])
                scores = line[2:]
                if line[0] not in vocab:
                    temp = self.ps.stem(line[0])
                    if temp not in vocab:
                        temp = estimate_subword(line[0], vocab, n_values)
                        if temp is None: continue 
                    line[0] = temp
                if line[1] not in vocab:
                    temp = self.ps.stem(line[1])
                    if temp not in vocab:
                        temp = estimate_subword(line[1], vocab, n_values)
                        if temp is None: continue 
                    line[1] = temp
                i1, i2 = vocab[line[0]] if line[0] in vocab else float(len(vocab)), vocab[line[1]] if line[1] in vocab else float(len(vocab))
                self.data.extend([((i1, i2), float(score)) for score in scores])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
    
