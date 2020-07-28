import pandas as pd
import itertools
import h5py

from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.initializers import Constant
from matplotlib import pyplot
import gensim
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt



hotel_review_df = pd.read_csv('data/deceptive-opinion.csv')
print(hotel_review_df)
# Finding the sentence with the most amount of words
text_length = {
    "lengths" : []
}
for sentence in hotel_review_df['text'].values:
  length = 0
  length = len(sentence.split())
  text_length['lengths'].append(length)
text_length_df = pd.DataFrame(text_length)
hotel_review_df = pd.concat([hotel_review_df, text_length_df], axis=1)

print(hotel_review_df)

print("Largest Word Count in Text: " + str(hotel_review_df['lengths'].max()))
print("Average Word Count in Text: " + str(hotel_review_df['lengths'].mean()))
print("Median Word Count in Text: " + str(hotel_review_df['lengths'].median()))

"""**3. Encoding Deceptive Labels**


*   Truthful: 1 
*   Deceptive: 0
"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

hotel_review_df['deceptive'] = le.fit_transform(hotel_review_df['deceptive'])

hotel_review_df

"""**4. Split Dataset (Training and Testing)**


*   Splitting Data 80/20 (Standard) - Can change train_size & test_size
"""

from sklearn.model_selection import train_test_split


train_Txt, test_Txt, train_y, test_y = train_test_split(hotel_review_df['text'].values, hotel_review_df['deceptive'].values,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=122)
print("Number of Training Records: " + str(train_y.shape[0]))
print("Number of Testing Records: " + str(test_y.shape[0]))

"""**5. Tokenize Text**"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Size of Vocab is 5000 (Common practice, can change)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_Txt)

train_X = tokenizer.texts_to_sequences(train_Txt)
test_X = tokenizer.texts_to_sequences(test_Txt)

# Add one due to 0 being reserved for padding
vocab_size = len(tokenizer.word_index) + 1

# Determined by the average of words in texts (Can change)
maxlen = 784

train_X = pad_sequences(train_X, padding='post', maxlen=maxlen)
test_X = pad_sequences(test_X, padding='post', maxlen=maxlen)

print(train_X)

"""**6. Pre-Trained Word Embedding and Early Stopping**"""

import numpy as np
from tqdm import tqdm
import codecs

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


glove_embedding_dim = 300
fasttext_embedding_dim = 300
word_embedding_dim = 300


embedding_matrix_glove = create_embedding_matrix('data/glove.6B.300d.txt', tokenizer.word_index,
                                                 glove_embedding_dim)
#
#
matrix_word = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
words_not_found = []
vocabulary_size=min(10000, len(tokenizer.word_index))
embedding_matrix_word = np.zeros((vocabulary_size, word_embedding_dim))
for word, i in tokenizer.word_index.items():
    if i>=vocabulary_size:
        continue
    try:
        embedding_vector = matrix_word[word]
        embedding_matrix_word[i] = embedding_vector
    except KeyError:
        embedding_matrix_word[i]=np.random.normal(0,np.sqrt(0.25),word_embedding_dim)


#
embeddings_index = {}
f = codecs.open('data/wiki-news-300d-1M.vec', encoding='utf-8')
for line in tqdm(f):
  values = line.rstrip().rsplit(' ')
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()
words_not_found = []
nb_words = min(10000, len(tokenizer.word_index))
embedding_matrix_fasttext = np.zeros((nb_words, fasttext_embedding_dim))
for word, i in tokenizer.word_index.items():
  if i >= nb_words:
      continue
  embedding_vector = embeddings_index.get(word)
  if (embedding_vector is not None) and len(embedding_vector) > 0:
      # words not found in embedding index will be all-zeros.
      embedding_matrix_fasttext[i] = embedding_vector
  else:
      words_not_found.append(word)


from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True, mode="min")

embedding_dim = 300
embedding_dim_glove = 300
embedding_dim_fasttext = 300
embedding_dim_word = 300

"""**6.1 Setup Functions for Precision, Recall, F1**"""

from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

"""**6.2 Attention Layer**"""

from keras.backend import dot, expand_dims, cast, tanh, exp, sum, floatx, epsilon, squeeze
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer

# TODO clean up code
# Code from https://www.kaggle.com/hsankesara/news-classification-using-han/notebook


def dot_product(x, kernel):
    return squeeze(dot(x, expand_dims(kernel)), axis=-1)



"""**8. RNN without WE**"""

from keras.models import Sequential
from keras import layers
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Bidirectional

vocab_size = len(tokenizer.word_index) + 1
print("vocab size: ")
print(vocab_size)


####################################################   BI-LSTM  #####################################################
# model = Sequential()
# model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
# model.add(layers.Embedding(vocab_size, embedding_dim_glove, input_length=maxlen, trainable=True, weights=[embedding_matrix_glove])),
# model.add(layers.Embedding(9309, embedding_dim_fasttext, input_length=maxlen, trainable=False, weights=[embedding_matrix_fasttext]))

model = Sequential([
  # layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
  layers.Embedding(9309, embedding_dim_word, input_length=maxlen, trainable=True, weights=[embedding_matrix_word]),

  # layers.Embedding(vocab_size, embedding_dim_glove, input_length=maxlen, trainable=True, weights=[embedding_matrix_glove]),
  # layers.Embedding(9309, embedding_dim_fasttext, input_length=maxlen, trainable=True, weights=[embedding_matrix_fasttext]),

  layers.Conv1D(64, 5, activation='relu', padding='valid', strides=1),
  Dropout(0.5),
  layers.MaxPooling1D(4),
  # Bidirectional(LSTM(64)),
  Bidirectional(LSTM(128)),
  Dropout(0.5),
  Dense(64, activation='relu'),
  Dropout(0.5),
  Dense(1, activation='sigmoid')
])

model.summary()
model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision_m, recall_m, f1_m])

history = model.fit(train_X, train_y,
                    epochs=20, validation_split=0.2, callbacks=[early_stopping],
                    batch_size=32)


# print(loss)

# history = model.fit(train_X, train_y,
#                     epochs=20,
#                     callbacks=[early_stopping],
#                     validation_data=(test_X, test_y),
#                     batch_size=32)
#

pyplot.title('Model Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['train', 'val'], loc='upper left')
# plt.show()
plt.savefig("cbiglove1/loss.png")

plt.title('Model Accuracy')
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['train', 'val'], loc='upper left')
# pyplot.show()
plt.savefig("cbiglove1/accu.png")


accuracy = model.evaluate(test_X, test_y, verbose=0)
print(accuracy)


# save model to single file
# model.save('cbiglove1/model.h5')
# model.save('cbilstm_model_trainword.h5')
# model.save('cbilstm_model_trainglove.h5')
# model.save('cbilstm_model_trainfasttxt.h5')
# model.save('clstmnopretrained.h5')
# model.save('lstm_model_trainfasttxt.h5')
# model.save('lstm_model_trainfasttxt.h5')


# [0.49369199126958846, 0.875, 0.9329925775527954, 0.8196017622947693, 0.8683851897716522]
# [[150  10]
#  [ 30 130]]


# from keras.models import load_model
from keras.models import load_model
# load model from single file
# model = load_model('lstm_model_train.h5')
train_y_predict = model.predict_classes(train_X)

test_y_predict = model.predict_classes(test_X)

# print(test_y_predict)

# Making the Confusion Matrix
# cm = confusion_matrix(train_y, train_y_predict) # Calulate Confusion matrix for train set.

cm2 = confusion_matrix(test_y, test_y_predict) # Calulate Confusion matrix for test set.

print(cm2)


# predict probabilities
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(test_y))]
lr_probs = model.predict_proba(test_X)

# print(lr_probs)
# keep probabilities for the positive outcome only
# lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(test_y, ns_probs)
lr_auc = roc_auc_score(test_y, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('CNN-BiLSTM: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(test_y, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(test_y, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='CNN-BiLSTM')
pyplot.title("Receiver Operating Characteristic")
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
# pyplot.show()
pyplot.savefig("cbiglove1/curve.png")



#
# pyplot.plot(history.history['acc'])
# pyplot.plot(history.history['val_acc'])
# pyplot.title('model accuracy')
# pyplot.ylabel('accuracy')
# pyplot.xlabel('epoch')
# pyplot.legend(['train', 'val'], loc='upper left')
# pyplot.show()
#
#
# pyplot.plot(history.history['loss'])
# pyplot.plot(history.history['val_loss'])
# pyplot.title('model loss')
# pyplot.ylabel('loss')
# pyplot.xlabel('epoch')
# pyplot.legend(['train', 'val'], loc='upper left')
# pyplot.show()

#no embedding
# [0.3559387758374214, 0.846875, 0.8930090308189392, 0.7884444475173951, 0.8345756769180298]
# [[145  15]
#  [ 34 126]]
# No Skill: ROC AUC=0.500
# BiLSTM: ROC AUC=0.932



#glove
# Epoch 00014: early stopping
# [0.4028843015432358, 0.81875, 0.8710943281650543, 0.7625256419181824, 0.8039073467254638]
# [[141  19]
#  [ 39 121]]
# No Skill: ROC AUC=0.500
# BiLSTM: ROC AUC=0.919

#fast
# [0.3867971181869507, 0.8375, 0.828255045413971, 0.839449018239975, 0.8312088906764984]
# # [[133  27]
# #  [ 25 135]]
# # No Skill: ROC AUC=0.500
# # BiLSTM: ROC AUC=0.911






#latest-glove
# Restoring model weights from the end of the best epoch
# Epoch 00018: early stopping
# [0.4122168004512787, 0.853125, 0.8484911620616913, 0.869636458158493, 0.8538979172706604]
# [[135  25]
#  [ 22 138]]
# No Skill: ROC AUC=0.500
# CNN-BiLSTM: ROC AUC=0.925



#fast2:

# Epoch 00013: early stopping
# [0.3559890627861023, 0.875, 0.8617237687110901, 0.8750127375125885, 0.8666243255138397]
# [[139  21]
#  [ 19 141]]
# No Skill: ROC AUC=0.500
# CNN-BiLSTM: ROC AUC=0.929

#noembed final:
# [0.3463210701942444, 0.85, 0.8959628820419312, 0.800266969203949, 0.8391462922096252]
# [[145  15]
#  [ 33 127]]
# No Skill: ROC AUC=0.500
# CNN-BiLSTM: ROC AUC=0.939


# 2020, fasttext
# [0.31872850507497785, 0.884375]
# [[139  21]
#  [ 16 144]]