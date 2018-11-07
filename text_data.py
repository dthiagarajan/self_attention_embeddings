from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Text data used for training word2vec model."""

    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(training_data)

    def __getitem__(self, idx):
        return training_data[idx]
