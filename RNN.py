import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, LSTM, Dropout, Activation

df = pd.read_csv('Sample_Data.csv')
train, test = train_test_split(df, test_size=0.2, random_state=42)
x_train = np.array(train.iloc[:,0])
y_train = np.array(train.iloc[:,1])
x_test = np.array(test.iloc[:,0])
y_test = np.array(test.iloc[:,1])

#Converts the O/P labels to one-hot encoding
y_train_hot = np.delete(to_categorical(y_train), 0, axis=1)
y_test_hot = np.delete(to_categorical(y_test, 6), 0, axis=1)

#Finds maximum number of words in any sentence of any row for padding
def max_words(array):
    l = 0
    for i in range(array.shape[0]):
        list = array[i].lower().split()
        if len(list)>l:
            l = len(list)
        else:
            continue
    return l

l = max_words(np.array(df.iloc[:, 0]))

#Generate a frequency dictionary of most occurring relevant words
def count_frequency(array):
    count={'WORDS':'COUNTS'}
    for i in range(array.shape[0]):
        list=array[i].lower().split()
        for j in range(len(list)):
            if list[j] in count:
                count[list[j]]+=1
            else:
                count[list[j]]=1
    new_keys=['excellent','love','good','average','satisfactory','timely','slow','poor','bad','trash','cheap','lag','price',
              'pathetic','fragile','mediocre','improve',]
    count_new={key:count[key] for key in count if key in new_keys}
    return count_new   

count_new=count_frequency(np.array(df.iloc[:,0]))

table=pd.Series(count_new,index=count_new.keys())
print(table)
table.plot.bar()
plt.xticks(rotation=-65)

#Extracts word2vec and word2index from GloVe
def load_GloVe(File):
    f = open(File, 'r')
    word2vec = {}
    word2index = {}
    for line in f:
        break_lines = line.split()
        word = break_lines[0]
        coeff = np.array([float(val) for val in break_lines[1:]])
        word2vec[word] = coeff
    f.close()
    t=1
    for key in word2vec:
        word2index[key]=t
        t+=1
    return word2vec, word2index

word2vec, word2index = load_GloVe('glove.6B.50d.txt')

#Regularises the I/P features into an array of dimension [(I/P).shape[0],l]
def regularise(X, word2index, l):
    A = np.zeros((np.shape(X)[0], l))
    for i in range(len(X)):
        list = X[i].lower().split()
        j = 0
        for h in list:
            A[i,j] = word2index[h]
            j+= 1
    return A

#Pre-trains and configures the embedding layer
def set_embedding_layer(word2index, word2vec):
    len_vocab = len(word2index) + 1
    embd_dim = 50 #Because using 50-dimensional GloVe vector
    embd_array = np.zeros((len_vocab, embd_dim))
    for word,index in word2index.items():
        embd_array[index, :] = word2vec[word]
    embedding_layer = Embedding(len_vocab, embd_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embd_array])
    return embedding_layer

def Sentiment_model(input_shape, word2vec, word2index):
    sentence_indices = Input(input_shape, dtype='int32')
    embedding_layer = set_embedding_layer(word2index, word2vec)
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(60, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(60, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation('softmax')(X)
    model = Model(inputs=sentence_indices, outputs=X)
    return model

model = Sentiment_model((l,), word2vec, word2index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_tr = regularise(x_train, word2index, l)
X_te = regularise(x_test, word2index, l)
model.fit(X_tr, y_train_hot, epochs = 50, batch_size = 19, shuffle=True)
loss, acc = model.evaluate(X_te, y_test_hot)
print("Test accuracy = ", acc)
