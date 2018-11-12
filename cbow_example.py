import os
import pickle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import *
from neg_loss import *
from text_data import *
from utils import *

from torch.utils.data import DataLoader


progressbar.streams.wrap_stderr()

CONTEXT_SIZE = 2
EMBEDDING_DIM = 50
NUM_EPOCHS = 40
BATCH_SIZE = 256
NEGATIVE_SAMPLING = False
USE_CUDA = True
PRE_LOADED = True

filename = "/scratch/datasets/large_text.txt"
if PRE_LOADED:
    vocab = pickle.load(open('/scratch/datasets/vocab.pkl', 'rb'))
    training_data = pickle.load(open('/scratch/datasets/training_data.pkl', 'rb'))
    word_to_ix = pickle.load(open('/scratch/datasets/word_to_ix.pkl', 'rb'))
    ix_to_word = pickle.load(open('/scratch/datasets/ix_to_word.pkl', 'rb'))
else:
    print("Parsing text and loading training data...")
    processed_text, vocab, word_to_ix, ix_to_word, training_data = load_data(filename,
                                                                CONTEXT_SIZE, model_type="cbow", subsampling=True, sampling_rate=0.001)

dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

losses = []
if NEGATIVE_SAMPLING: 
    loss_function = NEGLoss(ix_to_word, vocab)
else:
    loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

cuda = (torch.cuda.is_available() and USE_CUDA)
if cuda: 
    print("Using CUDA", flush=True)
    model.cuda()
Tensor = torch.cuda.LongTensor if cuda else torch.LongTensor
LossTensor = torch.cuda.FloatTensor if cuda else torch.Tensor 
print("Starting training", flush=True)
for epoch in range(NUM_EPOCHS):
    total_loss = LossTensor([0])
    print("Beginning epoch %d" % epoch, flush=True)
    progress_bar = progressbar.ProgressBar()
    for context, target in progress_bar(dataloader):
        context, target = [t.cuda() for t in context], [t.cuda() for t in target]
        context_var = autograd.Variable(Tensor(torch.stack(context)))
        focus_var = autograd.Variable(Tensor(torch.stack(target)))
        model.zero_grad()
        log_probs = model(context_var, focus_var)
        loss = loss_function(log_probs, focus_var.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    print("Epoch %d Loss: %.5f" % (epoch, total_loss[0]), flush=True)
    torch.save(model.state_dict(), open('/scratch/datasets/models/self_attention_embedding_model_%d.pt' % epoch, 'wb'))
    losses.append(total_loss)
torch.save(model.state_dict(), open('/scratch/datasets/models/self_attention_embedding_model.pt', 'wb'))

# Visualize embeddings
if EMBEDDING_DIM == 2:
    indices = np.random.choice(np.arange(len(vocab)), size=10, replace=False)
    for ind in indices:
        word = list(vocab.keys())[ind]
        input = autograd.Variable(Tensor([word_to_ix[word]]))
        vec = model.embeddings(input).data[0]
        x, y = vec[0], vec[1]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    plt.savefig("w2v.png")

