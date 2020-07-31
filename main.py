class Dataset:
    def __init__(self, path):

        self.df = pd.read_csv("training_data.csv")
        self.data = self.df.to_numpy()



        self.stopwords = set(stopwords.words('english'))
        self.essay_id = self.data[:,0]
        self.text = self.data[:,1]
        self.scores = self.data[:,2:8]
        self.new_data = []
        self.vocab = set()
        self.word_to_id = None

    def preprocess(self):
        for i in range(len(self.essay_id)):
            text = self.text[i].lower()
            text = " ".join([word for word in text.split() if '@' not in word])
            text = word_tokenize(text)
            text = [word for word in text if word not in self.stopwords]
            self.text[i] = text 

    def create_vocab(self):
        for line in self.text:
            for word in line:
                self.vocab.add(word)

        self.vocab = sorted(list(self.vocab))
        self.word_to_id = {word:i for i, word in enumerate(self.vocab)}

    def create_chunks(self):
        for idx in range(len(self.essay_id)):
            ess = self.text[idx]
            n = len(ess)
            self.new_data.append([ess[:n//3], self.scores[idx]])
            self.new_data.append([ess[n//3:2*n//3], self.scores[idx]])
            self.new_data.append([ess[2*n//3:], self.scores[idx]])
        self.new_data = np.array(self.new_data)

    def text_num(self):
        for i, line in enumerate(self.text):
            x = []
            for word in line:
                x.append(self.word_to_id[word])
            self.text[i] = x


import pandas as pd
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np


dataset = Dataset("./training_data.csv")

dataset.preprocess()
dataset.create_vocab()
print(len(dataset.vocab))
print(dataset.vocab[:10])
print(dataset.word_to_id['!'])
dataset.create_chunks()
dataset.text_num()
print(dataset.text[0])
"""
print(len(dataset.new_data), len(dataset.essay_id))
print("0: ", dataset.new_data[0, 0])
print("1: ", dataset.new_data[1, 0])
print("2: ", dataset.new_data[2, 0])
print(dataset.text[0])
"""
