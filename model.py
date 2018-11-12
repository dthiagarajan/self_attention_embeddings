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
        context_embeds = self.embeddings(inputs).permute(1, 0, 2)
        focus_embeds = self.embeddings(focus).permute(1, 2, 0)
        alpha = torch.matmul(torch.matmul(context_embeds, self.kernel.unsqueeze(0)), focus_embeds).permute(0, 2, 1)
        alpha = F.softmax(alpha, dim=-1)
        embeds = torch.matmul(alpha, context_embeds)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=-1).squeeze(1)
        return log_probs


