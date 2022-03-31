TRAINING_DATA_PATH = "data.pickle"
TESTING_DATA_PATH = "data.pickle"
VALIDATION_DATA_PATH = "data.pickle"
EMBEDDING_MODEL_PATH = "/home/elmo/"
NER_MODEL_PATH = "ner_elmo_model"
BATCH_SIZE = 32
MAX_LEN = 40
N_TAGS = 2
EPOCHS = 2

import pandas as pd
import pickle
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import (
    LSTM,
    Embedding,
    Dense,
    TimeDistributed,
    Dropout,
    Bidirectional,
    Lambda,
)
from keras.callbacks import ModelCheckpoint
from mlflow import log_metric, log_param, log_artifact
import mlflow
import datetime

print("Loading training data from disk.")


def data_reader(path):
    with open(path, "rb") as f:
        X, y = pickle.load(f)
    return X, y


def ElmoEmbedding(x):
    elmo_model = hub.Module(EMBEDDING_MODEL_PATH, trainable=True)
    return elmo_model(
        inputs={
            "tokens": tf.squeeze(tf.cast(x, "string")),
            "sequence_len": tf.constant(BATCH_SIZE * [MAX_LEN]),
        },
        signature="tokens",
        as_dict=True,
    )["elmo"]


def ner_model_definition():
    input_text = Input(shape=(MAX_LEN,), dtype="string")
    embedding = Lambda(ElmoEmbedding, output_shape=(MAX_LEN, 1024))(input_text)
    x = Bidirectional(
        LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)
    )(embedding)
    x_rnn = Bidirectional(
        LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)
    )(x)
    x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(N_TAGS, activation="softmax"))(x)
    model = Model(input_text, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def mlflow_handle_ner(X_tr, X_val):
    log_param("batch_size", BATCH_SIZE)
    log_param("epochs", EPOCHS)
    log_param("train_size", len(X_tr))
    log_param("val_size", len(X_val))


def main():
    X_tr, y_tr = data_reader(TRAINING_DATA_PATH)
    X_te, y_te = data_reader(TESTING_DATA_PATH)
    X_val, y_val = data_reader(VALIDATION_DATA_PATH)
    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print("Defining model architecture")
    model = ner_model_definition()
    y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
    mlflow.set_experiment("NER_ELMO")
    with mlflow.start_run():
        mlflow_handle_ner(X_tr, X_val)
        filepath = NER_MODEL_PATH + str(datetime.date.today()) + ".h5"
        checkpoint = ModelCheckpoint(
            filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )
        callbacks_list = [checkpoint]
        history = model.fit(
            np.array(X_tr),
            y_tr,
            validation_data=(np.array(X_val), y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks_list,
        )
        for i in history.history["loss"]:
            log_metric("loss", i)
        for i in history.history["val_loss"]:
            log_metric("val_loss", i)
        model.save_weights(NER_MODEL_PATH + str(datetime.date.today()) + ".h5")


if __name__ == "__main__":
    main()
