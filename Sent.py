from nltk.stem import PorterStemmer
from torch.utils.data import Dataset, DataLoader
import pickle as pkl


''' 
0 for negative, 1 for positive.
'''

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


class Sent(Dataset):
    def __init__(self, filename, vocab, vocab_idx, n_values=range(3, 7)):
        self.ps = PorterStemmer()
        self.data = []
        with open(filename, "r") as f:
            for line in f:
                inds = []  # indices in vocab of each word in this review
                label, text = line.replace('\n', '').split('\t')
                text = text.split(' ')
                for word in text:
                    if word not in vocab:
                        temp = self.ps.stem(word)
                        if temp not in vocab:
                            temp = estimate_subword(word, vocab, n_values)
                            if temp is None:
                                continue
                        word = temp
                    ind = vocab_idx[word]
                    assert ind < len(vocab)
                    inds.append(ind)
                self.data.append((inds, float(label))) # tuples of ([list of indices], label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    raw = '/home/dt372/self_attention_embeddings/data/moview_review_data_polarity/review_polarity_test.txt'
    vb = pkl.load(open('/share/nikola/export/dt372/vocab.pkl', 'rb'))
    vbidx = pkl.load(open('/share/nikola/export/dt372/word_to_ix.pkl', 'rb'))
    ds = Sent(raw,vb,vbidx)
    print(ds[4])
if __name__ == '__main__':
    main()
