from nltk.stem import PorterStemmer
from torch.utils.data import Dataset, DataLoader
import pickle as pkl


def estimate_subword(word, vocab, n_values):
    min_word, min_freq = None, len(vocab)
    for n in n_values:
        for i in range(len(word) - n + 1):
            subword = word[i:i+n]
            if subword in vocab:
                # freqs
                if vocab[subword] < min_freq:
                    min_word, min_freq = subword, vocab[subword]
    return min_word


class NER(Dataset):
    def __init__(self, filename, vocab, vocab_idx, n_values=range(3, 7)):
        self.ps = PorterStemmer()
        self.data = []
        with open(filename, 'r') as f:
            skip = 0
            for line in f:
                skip += 1
                if skip == 1:
                    continue
                if line == '\n':
                    continue
                n_line = line.replace('\n', '').split(' ')
                word, POS, chunk, entity = n_line
                if word not in vocab:
                    temp = self.ps.stem(word)
                    if temp not in vocab:
                        temp = estimate_subword(word, vocab, n_values)
                        if temp is None:
                            continue
                    word = temp
                ind = vocab_idx[word]
                assert ind < len(vocab)
                # tuple of (word, POS, chunk, entity)
                self.data.append((ind,POS,chunk,entity))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #returns tuple of (word, POS, chunk, entity) at idx in the data
        return self.data[idx]

# def main():
#     raw = '/Users/Amar/Downloads/wordsim353/NER/train.txt'
#     vb = pkl.load(open('/Users/Amar/Downloads/wordsim353/vocab.pkl', 'rb'))
#     vbidx = pkl.load(open('/Users/Amar/Downloads/wordsim353/word_to_ix.pkl', 'rb'))
#     ds = NER(raw,vb,vbidx)
#     print(ds.__getitem__(4))
# if __name__ == '__main__':
#     main()
