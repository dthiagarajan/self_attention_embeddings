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
EMBEDDING_DIMS = [50, 100, 200]
EMBEDDING_TYPES = ['self_attention', 'word2vec']
NUM_EPOCHS = 5
CUTOFFS = [1, 2, 3]
BATCH_SIZE = 1
USE_CUDA = True
DATA_PATH = "/scratch/datasets/wsim_data.csv"
VOCAB = pickle.load(open('/share/nikola/export/dt372/vocab.pkl', 'rb'))
VOCAB_IDX = pickle.load(open('/share/nikola/export/dt372/word_to_ix.pkl', 'rb'))
training_data = WordSimDataset(DATA_PATH, VOCAB, VOCAB_IDX)
dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
all_losses = {}
for EMBEDDING_DIM in EMBEDDING_DIMS:
    for EMBEDDING_TYPE in EMBEDDING_TYPES:
        for CUTOFF in CUTOFFS:
            model_descr = "%s_embedding_model_%d_%d" % (EMBEDDING_TYPE, EMBEDDING_DIM, CUTOFF - 1)
            print("Running model %s_embedding_model_%d_%d.pt" % (EMBEDDING_TYPE, EMBEDDING_DIM, CUTOFF - 1))
            MODEL_PATH = "/share/nikola/export/dt372/%s_embedding_model_%d_%d.pt" % (EMBEDDING_TYPE, EMBEDDING_DIM, CUTOFF - 1)
            model = WordSimRegressor(EMBEDDING_TYPE, MODEL_PATH)

            losses = []
            loss_function = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001)

            cuda = (torch.cuda.is_available() and USE_CUDA)
            if cuda: 
                print("Using CUDA", flush=True)
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
            out_file = '/share/nikola/export/dt372/wsim_%s.pt' % (model_descr)
            os.system('touch %s' % out_file)
            torch.save(model.state_dict(), open(out_file, 'wb'))
            progress_bar = progressbar.ProgressBar()
            dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=1)
            model.eval()
            total_loss = 0.
            print("Starting evaluation", flush=True)
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
            print("For %s, loss is %.5f" % (model_descr, total_loss))
            all_losses[model_descr] = total_loss
for k, v in all_losses.items():
    print("%s: %.5f" % (k, v))
