import pandas as pd
import numpy as np
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import gc
import multiprocessing
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.models import Sequential, Model
from keras.layers import concatenate, InputLayer, Input, Dense, LSTM, Embedding, Dropout, Bidirectional, Flatten, Reshape
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.regularizers import Regularizer
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import *
from keras.layers.merge import Add, Dot, Concatenate
print("import customized Attention implementations") 
print("Reference Link: https://www.kaggle.com/takuok/bidirectional-lstm-and-attention-lb-0-043")
import Attention

import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
import pydot

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from pymongo import MongoClient
conn_string = "mongodb://localhost:27017/"
try:
    db_client = MongoClient(conn_string)
    yelp_db = db_client.yelp
except:
    print("Cannot connect to MongoDB.")

# Function to load pre-trained GLoVE model
def load_glove_model(glove_file):
    print("Loading Glove Model")
    f = open(glove_file,'r', encoding="utf8")
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def shuffle_dframe(dataframe):
    temp = (dataframe.index.values).copy()
    np.random.shuffle(temp)
    dataframe = dataframe.set_index(temp)
    del temp
    dataframe = dataframe.sort_index()
    dataframe = dataframe.reset_index(drop=True)
    return dataframe

def seq_to_num(series, tokenizer):
    max_sent_len = 60
    temp_sentences = [" ".join(x[:max_sent_len]) for x in series]
    temp_sentences = pad_sequences(tokenizer.texts_to_sequences(temp_sentences), maxlen=max_sent_len, padding="post", truncating="post")
    return temp_sentences

# Loading word embeddings in 50, 100 and 200 dimensions
EMBED = load_glove_model("glove.6B.50d.txt")


## Sampling data fom Yelp database using MongoDB queries
# reviews = pd.DataFrame(yelp_db.user.aggregate([{"$sample":{"size": 500000}}]))
# business_data = pd.DataFrame(list(busin.find({"business_id": {"$in": list(test.iloc[:, 3])}})))
reviews = pd.read_csv("reviews.csv") # these reviews are ramdomnly sampled over the star ratings
business_data = pd.read_csv("business_data.csv")

reviews.drop(labels=['Unnamed: 0', 'Unnamed: 0.1', '_id', 'cool', 'date', 'funny',  'useful'], axis=1, inplace=True)
business_data.drop(labels=['Unnamed: 0', '_id', 'address', 'attributes', 'city', 'hours', 'is_open', 'latitude', 'longitude', 'name', 'postal_code', 'review_count', 'state'], axis=1, inplace=True)
reviews.hist(column="stars")
business_data.hist(column="stars")
reviews.rename(columns={"stars":"stars_users"}, inplace=True)
business_data.rename(columns={"stars":"stars_business"}, inplace=True)
merged_frame = pd.merge(reviews,business_data, on="business_id")

print("No. of users in the dataset:", len(merged_frame.user_id.unique()))
print("No. of businesses being reviews by the users:", len(merged_frame.business_id.unique()))

del reviews, business_data

merged_frame["text"] = merged_frame["text"].apply(lambda x: x.split())
merged_frame["categories"] = merged_frame["categories"].apply(lambda x: str(x).split(','))
merged_frame = shuffle_dframe(merged_frame)

# Building feature set for recommendations
merged_frame["recommend_user"] = np.where(merged_frame["stars_users"] >= 4, 1, 0)
merged_frame["recommend_business"] = np.where(merged_frame["stars_business"] >= 4, 1, 0)

# Splitting into train, validation and test sets
train = merged_frame.iloc[:round(len(merged_frame)*0.7), :]
val = merged_frame.iloc[round(len(merged_frame)*0.7):round(len(merged_frame)*0.85), :]
test = merged_frame.iloc[round(len(merged_frame)*0.85):, :]

Y_train_u = to_categorical(train.recommend_user, 4)
Y_val_u = to_categorical(val.recommend_user, 4)
Y_test_u = to_categorical(test.recommend_user, 4)
Y_train_b = to_categorical(train.recommend_business, 4)
Y_val_b = to_categorical(val.recommend_business, 4)
Y_test_b = to_categorical(test.recommend_business, 4)
gc.collect()

## setting text features
X_train_u_text = train["text"]
X_val_u_text = val["text"]
X_test_u_text = test["text"]

X_train_b_text = train["categories"]
X_val_b_text = val["categories"]
X_test_b_text = test["categories"]

# setting star features
X_train_u_stars = train["stars_users"]
X_val_u_stars = val["stars_users"]
X_test_u_stars = test["stars_users"]

X_train_b_stars = train["stars_business"]
X_val_b_stars = val["stars_business"]
X_test_b_stars = test["stars_business"]

del train, val, test
gc.collect()

tokenizer = Tokenizer(num_words=len(EMBED))
tokenizer.fit_on_texts(X_train_u_text)
X_train_u_text = seq_to_num(X_train_u_text, tokenizer)
X_val_u_text = seq_to_num(X_val_u_text, tokenizer)
X_test_u_text = seq_to_num(X_test_u_text, tokenizer)

X_train_b_text = seq_to_num(X_train_b_text, tokenizer)
X_val_b_text = seq_to_num(X_val_b_text, tokenizer)
X_test_b_text = seq_to_num(X_test_b_text, tokenizer)

vocab_weights = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, len(EMBED["hello"]))) # +1 is because the matrix indices start with 0

for word, i in tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
    try:
        embeddings_vector = EMBED[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        vocab_weights[i] = embeddings_vector

len(vocab_weights)
len(tokenizer.word_index)

embedding_in_dim = vocab_weights.shape[0]
embedding_out_dim = vocab_weights.shape[1]

l2_factor = [0.0001, 0.001, 0.005, 0.01, 0.05] # L2 penalty factor
drop_rate = [0.2, 0.3, 0.4] # rate for dropout
hidden_activation = ["relu", "sigmoid", "tanh"] # activation function for hidden layer
out_activation = ["softmax", "tanh"] # activation function for output layer
optimizer = ['adam']

print("###Model 1 - Two LSTM models concatenated to learn parallely###")
###Model 1 - Two LSTM models concatenated to learn parallely###
input_user_text = Input(shape=(60,))
model_user_text = Embedding(input_dim=int(embedding_in_dim), output_dim=int(embedding_out_dim), input_length=60, weights=[vocab_weights], trainable=False, name='user_text_embedding_layer')(input_user_text)
model_user_text = LSTM(32, return_sequences=False, name='lstm_layer1')(model_user_text)#Flatten(name='flatten_layer1')(model_user_text)
model_user_text = Dropout(rate=drop_rate[0], name='dropout_layer1')(model_user_text)
model_user_text = Dense(128, activation=hidden_activation[0], activity_regularizer=regularizers.l2(l2_factor[0]), name='user_text_hidden_layer')(model_user_text)
model_user_text = Dense(4, activation=out_activation[0], name = "user_text_output_layer")(model_user_text)

input_business_text = Input(shape=(60,), name = "business_text_input")
model_business_text = Embedding(input_dim=int(embedding_in_dim), output_dim=int(embedding_out_dim), input_length=60, weights=[vocab_weights], trainable=False, name='business_text_embedding_layer')(input_business_text)
model_business_text = Flatten(name='flatten_layer2')(model_business_text)#LSTM(32, return_sequences=False, name='lstm_layer2')(model_business_text)
model_business_text = Dropout(rate=drop_rate[0], name='dropout_layer2')(model_business_text)
model_business_text = Dense(128, activation=hidden_activation[0], activity_regularizer=regularizers.l2(l2_factor[0]), name='business_text_hidden_layer')(model_business_text)
model_business_text = Dense(4, activation=out_activation[1], name = "business_text_output_layer")(model_business_text)

merged_model = concatenate([model_user_text, model_business_text])
merged_model = Dense(4, activation="relu")(merged_model)
dotprod = Dot(axes=1)([model_user_text, model_business_text])
output = Add()([merged_model, dotprod])
model = Model(inputs=[input_user_text, input_business_text], outputs=[output])
model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])
###End Model 1###

os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import Image

plot_model(model, to_file='model1.png', show_layer_names=True, show_shapes=True)
Image('encoder_model.png')

train_result = model.fit([X_train_u_text, X_train_b_text], Y_train_u, batch_size=1024, epochs=5, validation_data=([X_val_u_text, X_val_b_text], Y_val_u))

# plot history
plt.plot(train_result.history['loss'], label="training", color="blue")
plt.plot(train_result.history['val_loss'], label="validation", color="red")
plt.legend(loc="right")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model 1 -Two LSTM models concatenated to learn parallel")
plt.show()

test_loss, test_mse, test_mae = model.evaluate([X_test_u_text, X_test_b_text], Y_test_u, batch_size=1024)
print("Test set MSE", round(test_mse, 4))

print("###Model 2 - Bi-LSTM model for User text concatenated to learn parallely with Business Features###")
###Model 2 - Bi-LSTM model for User text concatenated to learn parallely with Business Features###
input_user_text = Input(shape=(60,))
model_user_text = Embedding(input_dim=int(embedding_in_dim), output_dim=int(embedding_out_dim), input_length=60, weights=[vocab_weights], trainable=False, name='user_text_embedding_layer')(input_user_text)
model_user_text = Bidirectional(LSTM(32, return_sequences=False, name='lstm_layer1'))(model_user_text)#Flatten(name='flatten_layer1')(model_user_text)
model_user_text = Dropout(rate=drop_rate[0], name='dropout_layer1')(model_user_text)
model_user_text = Dense(128, activation=hidden_activation[0], activity_regularizer=regularizers.l2(l2_factor[0]), name='user_text_hidden_layer')(model_user_text)
model_user_text = Dense(4, activation=out_activation[0], name = "user_text_output_layer")(model_user_text)

input_business_text = Input(shape=(60,), name = "business_text_input")
model_business_text = Embedding(input_dim=int(embedding_in_dim), output_dim=int(embedding_out_dim), input_length=60, weights=[vocab_weights], trainable=False, name='business_text_embedding_layer')(input_business_text)
model_business_text = Flatten(name='flatten_layer2')(model_business_text)#LSTM(32, return_sequences=False, name='lstm_layer2')(model_business_text)
model_business_text = Dropout(rate=drop_rate[0], name='dropout_layer2')(model_business_text)
model_business_text = Dense(128, activation=hidden_activation[0], activity_regularizer=regularizers.l2(l2_factor[0]), name='business_text_hidden_layer')(model_business_text)
model_business_text = Dense(4, activation=out_activation[1], name = "business_text_output_layer")(model_business_text)

merged_model = concatenate([model_user_text, model_business_text])
merged_model = Dense(4, activation="relu")(merged_model)
dotprod = Dot(axes=1)([model_user_text, model_business_text])
output = Add()([merged_model, dotprod])
model = Model(inputs=[input_user_text, input_business_text], outputs=[output])
model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])
###End Model 2###

plot_model(model, to_file='model2.png', show_layer_names=True, show_shapes=True)
Image('encoder_model.png')

train_result = model.fit([X_train_u_text, X_train_b_text], Y_train_u, batch_size=1024, epochs=5, validation_data=([X_val_u_text, X_val_b_text], Y_val_u))

plt.plot(train_result.history['loss'], label="training", color="blue")
plt.plot(train_result.history['val_loss'], label="validation", color="red")
plt.legend(loc="right")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model 2 - Bi-LSTM model")
plt.show()

test_loss, test_mse, test_mae = model.evaluate([X_test_u_text, X_test_b_text], Y_test_u, batch_size=1024)
print("Test set MSE", round(test_mse, 4))


print("###Model 3 - Attention based Bi-LSTM model for User text concatenated to learn parallely with Business Features###")
###Model 2 - Attention based Bi-LSTM model for User text concatenated to learn parallely with Business Features###
input_user_text = Input(shape=(60,))
model_user_text = Embedding(input_dim=int(embedding_in_dim), output_dim=int(embedding_out_dim), input_length=60, weights=[vocab_weights], trainable=True, name='user_text_embedding_layer')(input_user_text)
model_user_text = Bidirectional(LSTM(32, return_sequences=True, name='lstm_layer1'))(model_user_text)#Flatten(name='flatten_layer1')(model_user_text)
model_user_text = Dropout(rate=drop_rate[0], name='dropout_layer1')(model_user_text)
model_user_text = Attention(60)(model_user_text)
model_user_text = Dense(128, activation=hidden_activation[0], activity_regularizer=regularizers.l2(l2_factor[0]), name='user_text_hidden_layer')(model_user_text)
model_user_text = Dense(4, activation=out_activation[0], name = "user_text_output_layer")(model_user_text)

input_business_text = Input(shape=(60,), name = "business_text_input")
model_business_text = Embedding(input_dim=int(embedding_in_dim), output_dim=int(embedding_out_dim), input_length=60, weights=[vocab_weights], trainable=False, name='business_text_embedding_layer')(input_business_text)
model_business_text = Flatten(name='flatten_layer2')(model_business_text)#LSTM(32, return_sequences=False, name='lstm_layer2')(model_business_text)
model_business_text = Dropout(rate=drop_rate[0], name='dropout_layer2')(model_business_text)
model_business_text = Dense(128, activation=hidden_activation[0], activity_regularizer=regularizers.l2(l2_factor[0]), name='business_text_hidden_layer')(model_business_text)
model_business_text = Dense(4, activation=out_activation[1], name = "business_text_output_layer")(model_business_text)

merged_model = concatenate([model_user_text, model_business_text])
merged_model = Dense(4, activation="relu")(merged_model)
dotprod = Dot(axes=1)([model_user_text, model_business_text])
output = Add()([merged_model, dotprod])
model = Model(inputs=[input_user_text, input_business_text], outputs=[output])
model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])
###End Model 3###

plot_model(model, to_file='model3.png', show_layer_names=True, show_shapes=True)
Image('encoder_model.png')

train_result = model.fit([X_train_u_text, X_train_b_text], Y_train_u, batch_size=1024, epochs=5, validation_data=([X_val_u_text, X_val_b_text], Y_val_u))

plt.plot(train_result.history['loss'], label="training", color="blue")
plt.plot(train_result.history['val_loss'], label="validation", color="red")
plt.legend(loc="right")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model 3 - Attention based Bi-LSTM model")
plt.show()

test_loss, test_mse, test_mae = model.evaluate([X_test_u_text, X_test_b_text], Y_test_u, batch_size=1024)
print("Test set MSE", round(test_mse, 4))
