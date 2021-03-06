import gensim
import os
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
class VanillaCBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(VanillaCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = torch.mean(self.embeddings(inputs), dim=0)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=-1).squeeze()
        return log_probs


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


class RareWordRegressor(nn.Module):
    def __init__(self, embedding_type, model_path, *args):
        super(RareWordRegressor, self).__init__()
        pretrained_model_params = torch.load(model_path)
        if 'embeddings.weight' in pretrained_model_params:
            self.embeddings = nn.Embedding.from_pretrained(pretrained_model_params['embeddings.weight'])
        elif 'module.embeddings.weight' in pretrained_model_params:
            self.embeddings = nn.Embedding.from_pretrained(pretrained_model_params['module.embeddings.weight'])
        self.embeddings.requires_grad = False
        self.fc = nn.Linear(self.embeddings.embedding_dim * 2, 1)

    def forward(self, input):
        embeds = self.embeddings(input)
        sim = torch.cat((embeds[:, 0, :], embeds[:, 1, :]), dim=1)
        out = self.fc(sim)
        out = F.relu(out).squeeze(1)
        return out

class WordSimRegressor(nn.Module):
    def __init__(self, embedding_type, model_path, *args):
        super(WordSimRegressor, self).__init__()
        pretrained_model_params = torch.load(model_path)
        if 'embeddings.weight' in pretrained_model_params:
            self.embeddings = nn.Embedding.from_pretrained(pretrained_model_params['embeddings.weight'])
        elif 'module.embeddings.weight' in pretrained_model_params:
            self.embeddings = nn.Embedding.from_pretrained(pretrained_model_params['module.embeddings.weight'])
        self.embeddings.requires_grad = False
        self.fc = nn.Linear(self.embeddings.embedding_dim * 2, 1)
    def forward(self, input):
        embeds = self.embeddings(input)
        sim = torch.cat((embeds[:, 0, :], embeds[:, 1, :]), dim=1)
        out = self.fc(sim)
        out = F.relu(out).squeeze(1)
        return out


class SentRegressor(nn.Module):
    def __init__(self, embedding_type, model_path, *args):
        super(SentRegressor, self).__init__()
        pretrained_model_params = torch.load(model_path)
        if 'embeddings.weight' in pretrained_model_params:
            self.embeddings = nn.Embedding.from_pretrained(
                pretrained_model_params['embeddings.weight'])
        elif 'module.embeddings.weight' in pretrained_model_params:
            self.embeddings = nn.Embedding.from_pretrained(
                pretrained_model_params['module.embeddings.weight'])
        self.embeddings.requires_grad = False
        self.fc = nn.Linear(self.embeddings.embedding_dim, 2)

    def forward(self, input):
        embeds = torch.mean(self.embeddings(input), dim=0)
        out = self.fc(embeds)
        out = F.softmax(out).squeeze(1)
        return out
