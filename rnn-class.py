# %%
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
# %%
headers = ['date', 'where', 'type', 'title']
theses_df =  pd.read_csv('res/theses.tsv', sep="\t", names=headers)
titles = []
types = []
for idx, row in theses_df.iterrows():
    if row['type'] in ['Bachelor', 'Master']:
        titles.append(row['title'])
        if(row['type'] == 'Bachelor'):
            types.append(0)
        else:
            types.append(1)

X_train, X_test, y_train, y_test = train_test_split(titles, types, test_size=0.2, random_state=42)

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))
# %%
VOCAB_SIZE = 10000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(titles)

len(encoder.get_vocabulary())

# %%
bi_lstm = tf.keras.Sequential([
    #tf.keras.Input(shape=(1,), dtype=tf.string),
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

bi_lstm.summary()

bi_lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = bi_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

bi_lstm.evaluate(X_test, y_test)

# %%

lstm = tf.keras.Sequential([
    #tf.keras.Input(shape=(1,), dtype=tf.string),
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

lstm.summary()

lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = lstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

lstm.evaluate(X_test, y_test)

# %%

gru = tf.keras.Sequential([
    #tf.keras.Input(shape=(1,), dtype=tf.string),
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

gru.summary()

gru.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = gru.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

gru.evaluate(X_test, y_test)

# %%
rnn = tf.keras.Sequential([
    #tf.keras.Input(shape=(1,), dtype=tf.string),
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.SimpleRNN(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

rnn.summary()

rnn.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = rnn.fit(X_train, y_train, epochs=7, validation_data=(X_test, y_test))

rnn.evaluate(X_test, y_test)
# %%
