import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
# create a dataframe using texts and lables
trainDF = train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/NLP/data/training_data.csv')
trainDF.head()

trainDF = trainDF[['text','organization']]
trainDF.columns = ['text','label']
trainDF.head()

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

max_len = 0
ind = 0
for t in train_x:
  if len(t.split()) > max_len:
    max_len = len(t.split())
max_len


# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=max_len)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=max_len)

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
train_seq_x1 = torch.from_numpy(train_seq_x).to(torch.int64).to(device)
train_y1 = torch.from_numpy(train_y).to(torch.int64).to(device)

valid_seq_x1 = torch.from_numpy(valid_seq_x).to(torch.int64).to(device)
valid_y1 = torch.from_numpy(valid_y).to(torch.int64).to(device)


batch_size = 50
train_loader = DataLoader(TensorDataset(train_seq_x1, train_y1), batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(TensorDataset(valid_seq_x1, valid_y1), batch_size = batch_size, shuffle = True)


train_seq_x1.shape, train_y1.shape




class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_num, output_num, layer_num):
      super().__init__()
      self.vocab_size = vocab_size
      self.layer_num = layer_num
      self.hidden_num = hidden_num

      self.embedding = nn.Embedding(vocab_size, embedding_size)
      self.lstm = nn.LSTM(embedding_size, hidden_num, layer_num)
      self.fc = nn.Linear(hidden_num, output_num)
      self.relu = nn.ReLU()

        
    def forward(self, word_seq):
      word_emb = self.embedding(word_seq)
      print('word_emb', word_emb.size())
      lstm_out,h = self.lstm(word_emb)
      print('lstm_out1', lstm_out.size())
      lstm_out = lstm_out.contiguous().view(-1, self.hidden_num)
      print('lstm_out2', lstm_out.size())
      fc_out = self.fc(lstm_out)
      print('fc_out', fc_out.size())
      relu_out = self.relu(fc_out)
      print('relu_out', relu_out.size())
      relu_out = relu_out.view(batch_size, -1) 
      relu_out = relu_out[:,-1]
      print('relu_out', relu_out.size())
      return relu_out, h



def train(data_loader, classifier, loss_function, optimizer):
    classifier.train()
    loss = 0
    losses = []
    
    accuracy = 0
    accuracies = []
    for i, (texts, labels) in enumerate(data_loader):
        if(texts.shape[0] != batch_size):
            break
        print(texts.size())
        print(labels.size())
        texts = texts.cuda()
        labels = torch.FloatTensor(labels)
        labels = labels.cuda()
        optimizer.zero_grad()
        predictions,h = classifier(texts)
        print(predictions.type(), labels.type())
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())        
    return losses.mean()


vocab_size = vocab_len
# n_vocab = len(embedding_matrix)
embedding_size = 300
hidden_num = 512
output_num = 1
layer_num = 2

rnn_model = LSTM_Model(vocab_size, embedding_size, hidden_num, output_num, layer_num)
rnn_model.cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(rnn_model.parameters(), lr=0.0001, momentum=0.9)
epochs = 5
for epoch in range(0, epochs):
    print("epoch:", epoch + 1)
    train(train_loader, rnn_model, loss_function, optimizer)
    #print("training_loss:", training_loss)

