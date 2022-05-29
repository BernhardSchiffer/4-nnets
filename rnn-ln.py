# %%
import torch
from torch import nn
import pandas as pd
from collections import Counter
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        headers = ['date', 'where', 'type', 'title']
        theses_df =  pd.read_csv('res/theses.tsv', sep="\t", names=headers)
        text = theses_df['title'].str.cat(sep=' <EOS> ')
        return text.split(' ')

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
        )

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state
        
    def init_state(self, sequence_length):

        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

# %%
def train(dataset, model, epochs, batch_size, sequence_length):
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        state_h, state_c = model.init_state(sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()
            #print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item()}')
        print(f'epoch: {epoch}, loss: {loss.item()}')

def predict(dataset, model, text, next_words=100):
    model.eval()
    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        word = dataset.index_to_word[word_index]
        if word == '<EOS>':
            break
        else:
            words.append(word)

    return words

# %%
dataset = Dataset(sequence_length=4)

print(dataset.__getitem__(5))
model = Model(dataset)
model.to(device)
# %%
train(dataset, model, epochs=40, batch_size=128, sequence_length=4)
# %%
print(predict(dataset, model, text='Konzeption'))
print(predict(dataset, model, text='Konzeption'))
print(predict(dataset, model, text='Konzeption'))
print(predict(dataset, model, text='Konzeption'))
# %%