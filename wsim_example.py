import os
import pickle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from WordSim import * 
from model import *
from rare_word_data import *
from torch.utils.data import DataLoader

progressbar.streams.wrap_stderr()

CONTEXT_SIZE = 2
EMBEDDING_DIM = 50
EMBEDDING_TYPE = 'word2vec'
NUM_EPOCHS = 5
BATCH_SIZE = 1
USE_CUDA = True
DATA_PATH = "/Users/Amar/Downloads/wordsim353/combined_data.csv"  # data file in memory
MODEL_PATH = "/Users/Amar/Downloads/checkpoints/self_attention_embedding_model_%d_%d.pt" % (EMBEDDING_DIM, NUM_EPOCHS - 1)  # model from email
VOCAB = pickle.load(open('/Users/Amar/Downloads/vocab.pkl', 'rb'))

if EMBEDDING_TYPE == 'word2vec':
    model = RareWordRegressor('word2vec')
else:
    embedding_model = CBOW(len(VOCAB), EMBEDDING_DIM, CONTEXT_SIZE)
    embedding_model.load_state_dict(torch.load(MODEL_PATH))
    model = RareWordRegressor('self_attention', embedding_model)

training_data = WordSimDataset(DATA_PATH, VOCAB)
dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

losses = []
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

cuda = (torch.cuda.is_available() and USE_CUDA)
if cuda: 
    print("Using CUDA", flush=True)
    model = nn.DataParallel(model)
    model.cuda()
model.train()
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LossTensor = torch.cuda.FloatTensor if cuda else torch.Tensor 
print("Starting training", flush=True)
for epoch in range(NUM_EPOCHS):
    total_loss = LossTensor([0])
    print("Beginning epoch %d" % epoch, flush=True)
    progress_bar = progressbar.ProgressBar()
    for input, target in progress_bar(dataloader):
        model.zero_grad()
        if cuda:
            input, target = [it.cuda() for it in input], [it.cuda() for it in target]
        input = autograd.Variable(LongTensor(torch.stack(input, dim=1).long()))
        target = autograd.Variable(FloatTensor(torch.stack(target).float()))
        preds = model(input)
        loss = loss_function(preds, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.data)
    print("Epoch %d Loss: %.5f" % (epoch, total_loss[0]), flush=True)
    losses.append(total_loss)

dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=1)
model.eval()
for input, target in progress_bar(dataloader):
    model.zero_grad()
    if cuda:
        input, target = [it.cuda() for it in input], [it.cuda() for it in target]
    input = autograd.Variable(LongTensor(torch.stack(input, dim=1).long()))
    target = autograd.Variable(FloatTensor(torch.stack(target).float()))
    preds = model(input)
    print(preds)
    print(target)
    print('----')
    loss = loss_function(preds, target)
    loss.backward()
    optimizer.step()
    total_loss += float(loss.data)
