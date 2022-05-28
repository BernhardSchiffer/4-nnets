# %%
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
from scipy.stats import rankdata
# %%
# load data
df = pd.read_csv('res/theses.tsv', sep='\t')
df.columns = ["year","type", "degree" ,"title"]
data = df["title"].tolist()
corpus = [x.lower() for x in data]

def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(corpus)

print(len(tokenized_corpus))
# %%
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)
print(vocabulary_size)
# %%
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
# %%
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 101
learning_rate = 0.01

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
# %%
def similarity(v):
    similarities = []
    for i in range(vocabulary_size):
        u = W2[i]
        sim = torch.dot(v,u)/(torch.norm(v)*torch.norm(u))
        similarities.append(sim.item())
    values = np.array(similarities)
    rankvalues = rankdata(values) # Ranks all data; 1 for smallest, n for biggest
    valuelist = list(rankvalues)
    closest = valuelist.index(9329.0) # vocabulary_size - 1 
    #closest = np.argmax(values)
    mostsim = values[closest]
    word = vocabulary[closest]
    return word, mostsim

# Most simular words
words = ['Konzeption', 'Cloud', 'virtuelle']

for w in words:
    print(f'similarity for {w}: {similarity(W2[word2idx[str.lower(w)]])}')

#similarity for Konzeption: ('konzept', 0.9865105152130127)
#similarity for Cloud: ('klassifikation', 0.9856687188148499)
#similarity for virtuelle: ('eindämmung', 0.9881880879402161)
# %%

import torch
import pandas as pd
from collections import Counter
class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]
    def load_words(self):
        df = pd.read_csv('res/theses.tsv', sep='\t')
        df.columns = ["year","type", "degree" ,"title"]
        data = df["title"].tolist()
        corpus = [x.lower() for x in data]
        
        train_df = pd.read_csv('data/reddit-cleanjokes.csv')
        text = train_df['Joke'].str.cat(sep=' ')
        return text.split(' ')
    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length
    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )
