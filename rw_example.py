import os
import pickle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import *
from rare_word_data import *
from torch.utils.data import DataLoader

progressbar.streams.wrap_stderr()

CONTEXT_SIZE = 2
EMBEDDING_DIM = 100
EMBEDDING_TYPE = 'self_attention'
# EMBEDDING_TYPE = 'word2vec'
NUM_EPOCHS = 5
BATCH_SIZE = 1
USE_CUDA = True
DATA_PATH = "/share/nikola/export/dt372/rw.txt"
MODEL_PATH = "/scratch/datasets/models/self_attention_embedding_model_%d_%d.pt" % (EMBEDDING_DIM, NUM_EPOCHS - 1)
VOCAB = pickle.load(open('/scratch/datasets/vocab.pkl', 'rb'))
cuda = (torch.cuda.is_available() and USE_CUDA)

print("Using %s to regress rare word similarity" % EMBEDDING_TYPE)
if EMBEDDING_TYPE == 'word2vec':
    model = RareWordRegressor('word2vec')
else:
    embedding_model = CBOW(len(VOCAB), EMBEDDING_DIM, CONTEXT_SIZE)
    if cuda:
        embedding_model = nn.DataParallel(embedding_model)
    embedding_model.load_state_dict(torch.load(MODEL_PATH))
    model = RareWordRegressor('self_attention', embedding_model)

training_data = RareWordDataset(DATA_PATH, VOCAB)
dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

losses = []
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

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
if EMBEDDING_TYPE == 'self_attention':
    out_file = "/scratch/datasets/models/rare_word_%s_%d_model.pt" % (EMBEDDING_TYPE, EMBEDDING_DIM)
elif EMBEDDING_TYPE == 'word2vec':
    out_file = '/scratch/datasets/models/rare_word_word2vec_300_model.pt'
os.system('touch %s' % out_file)
torch.save(model.state_dict(), open(out_file, 'wb'))

dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=1)
progress_bar = progressbar.ProgressBar()
model.eval()
total_loss = LossTensor([0])
for input, target in progress_bar(dataloader):
    model.zero_grad()
    if cuda:
        input, target = [it.cuda() for it in input], [it.cuda() for it in target]
    input = autograd.Variable(LongTensor(torch.stack(input, dim=1).long()))
    target = autograd.Variable(FloatTensor(torch.stack(target).float()))
    preds = model(input)
    loss = loss_function(preds, target)
    total_loss += float(loss.data)
print("Total Loss", total_loss)
