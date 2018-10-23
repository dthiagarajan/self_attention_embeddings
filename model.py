import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

''' Continuous bag-of-words model for self-attention word2vec.

Parameters:
    vocab_size: number of defined words in the vocab
    embedding_dim: desired embedded vector dimension
    context_size: number of context words used

'''
class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.kernel = nn.Parameter(torch.rand(embedding_dim, embedding_dim))
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs, focus):
        embeds = self.embeddings(inputs)
        alpha = torch.mm(torch.mm(embeds, self.kernel), self.embeddings(focus).t())
        alpha = F.softmax(alpha)
        embeds = torch.sum(torch.mul(alpha, embeds), 0, keepdim=True)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out)
        return log_probs


