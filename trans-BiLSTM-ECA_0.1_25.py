import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import re
from scipy import interp
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, multiply
from keras.layers import Flatten
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale
import utils.tools as utils
import math


# from tensorflow.keras.layers import Input


def to_class(p):
    return np.argmax(p, axis=1)


def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y


# Origanize data
def get_shuffle(dataset, label):
    # shuffle data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset, label


'''多头Attention'''


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


'''Transformer的Encoder部分'''


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


'''Transformer输入的编码层'''
# class TokenAndPositionEmbedding(layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim):
#         super(TokenAndPositionEmbedding, self).__init__()
#         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions

'''读取数据'''
data_ = pd.read_csv('SMOTE-RENN_170.csv')
data = np.array(data_)
data = data[:, 1:]
[m1, n1] = np.shape(data)
# label1=np.ones((int(m1/2),1))#Value can be changed
# label2=np.zeros((int(m1/2),1))
label1 = np.ones(((157696), 1))  # Value can be changed
label2 = np.zeros(((100737), 1))
label = np.append(label1, label2)
X_ = scale(data)
y_ = label
X, y = get_shuffle(X_, y_)
sepscores = []
sepscores_ = []
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5

vocab_size = 258434 * 2  # Only consider the top 20k words
maxlen = 170  # Only consider the first 200 words of each movie review

'''搭建模型'''
# embed_dim = 172  # Embedding size for each token
input_dim = 170
num_heads = 2  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer


##############################################################################


def ECA_block(input_dim, b=1, gamma=2, name=""):
    channel = input_dim.shape[-1]
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    x = layers.AvgPool1D(strides=1, padding="SAME", name="advage_pooling")(input_dim)

    # x = np.reshape([-1,1])(x)
    x = Conv1D(1, kernel_size=kernel_size, padding="same", name="eca_layer_" + str(name), use_bias=False,
               activation='sigmoid')(x)
    # x = np.reshape([1, 1, -1])(x)

    eca_out = multiply([input_dim, x])
    return eca_out


# model = keras.Sequential()
# model.add(layers.Input(shape=(maxlen,)))
# transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
# x = transformer_block(inputs)


inputs = keras.Input(shape=(None, maxlen))
# inputs = layers.Input(shape=(maxlen,))
# embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
# x = embedding_layer(inputs)
transformer_block = TransformerBlock(input_dim, num_heads, ff_dim)
x = transformer_block(inputs)
# x = layers.GlobalAveragePooling1D()(x)
# x = layers.Dropout(0.1)(x)
x = Bidirectional(LSTM(int(input_dim / 2), return_sequences=True))(x)
x = layers.Dropout(0.1)(x)
x = Bidirectional(LSTM(int(input_dim / 4), return_sequences=True))(x)
# x = Bidirectional(GRU(128,return_sequences=True))(x)
x = ECA_block(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(int(input_dim / 4), activation='relu')(x)
x = layers.Dense(int(input_dim / 8), activation="relu")(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

sepscores = []
probas_rnn = []
tprs_rnn = []
sepscore_rnn = []
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5
skf = StratifiedKFold(n_splits=5)
for train, test in skf.split(X, y):
    y_train = utils.to_categorical(y[train])  # generate the resonable results
    clf_rnn = model
    X_train_rnn = np.reshape(X[train], (-1, 1, input_dim))
    X_test_rnn = np.reshape(X[test], (-1, 1, input_dim))
    clf_list = clf_rnn.fit(X_train_rnn, to_categorical(y[train]).reshape((-1, 1, 2)), epochs=25)  # nb_epoch改为epochs
    y_rnn_probas = clf_rnn.predict(X_test_rnn)
    y_rnn_probas = y_rnn_probas.reshape((-1, 2))
    probas_rnn.append(y_rnn_probas)
    y_class = utils.categorical_probas_to_classes(y_rnn_probas)

    y_test = utils.to_categorical(y[test])  # generate the test
    ytest = np.vstack((ytest, y_test))
    y_test_tmp = y[test]
    yscore = np.vstack((yscore, y_rnn_probas))

    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class, y[test])
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(y[test], y_rnn_probas[:, 1])
    tprs_rnn.append(interp(mean_fpr, fpr, tpr))
    tprs_rnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    sepscore_rnn.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])

row = ytest.shape[0]
ytest = ytest[np.array(range(1, row)), :]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_54517.csv')

yscore_ = yscore[np.array(range(1, row)), :]
yscore_sum = pd.DataFrame(data=yscore_)
yscore_sum.to_csv('yscore_sum_54517.csv')

scores = np.array(sepscore_rnn)
result1 = np.mean(scores, axis=0)
H1 = result1.tolist()
sepscore_rnn.append(H1)
result = sepscore_rnn
# data_csv = pd.DataFrame(data=result)
# data_csv.to_csv('SE__train_GL_0.03.csv')

data_csv = pd.DataFrame(data=result)
colum = ['ACC', 'precision', 'npv', 'Sn', 'Sp', 'MCC', 'F1', 'AUC']
# ro=['1', '2', '3','4','5','6','7','8','9','10','11']
ro = ['1', '2', '3', '4', '5', '6']
data_csv = pd.DataFrame(columns=colum, data=result, index=ro)
data_csv.to_csv('trans_54517.csv')

