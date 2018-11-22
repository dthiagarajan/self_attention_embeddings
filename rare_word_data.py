from torch.utils.data import Dataset, DataLoader


class RareWordDataset(Dataset):
    def __init__(self, filename, vocab):
        self.data = []
        with open(filename, "r") as f:
            for line in [line.strip().split() for line in f]:
                word_pair = (line[0], line[1])
                scores = line[2:]
                self.data.extend([([vocab[line[0]], vocab[line[1]]], score) for score in scores])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
    
