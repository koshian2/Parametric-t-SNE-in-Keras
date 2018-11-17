from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import Callback
import os, shutil, zipfile, glob
import matplotlib.pyplot as plt

from tsne import TSNE

import numpy as np

def create_model():
    input = Input((784,))
    x = Dense(500, activation="relu")(input)
    x = Dense(500, activation="relu")(x)
    x = Dense(2000, activation="relu")(x)
    x = Dense(2)(x) # 活性化関数はなし

    model = Model(input, x)
    return model


class Sampling(Callback):
    def __init__(self, model, X, y):
        self.X = (X / 255.0).reshape(X.shape[0], -1)
        self.y = np.ravel(y)
        self.model = model

    def on_train_begin(self, logs):
        if os.path.exists("plots"):
            shutil.rmtree("plots")
        os.mkdir("plots")

    def on_epoch_end(self, epoch, logs):
        latent = self.model.predict(self.X)
        cmp = plt.get_cmap("Set1")
        plt.figure()
        for i in range(10):
            select_flag = self.y == i
            plt_latent = latent[select_flag, :]
            plt.scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(i), marker=f"${i}$")
        plt.savefig(f"plots/epochs{epoch:02}.png")

def generator(data_X, batch_size=5000):
    if data_X.ndim == 3:
        dims = data_X.shape[1] * data_X.shape[2]
    elif data_X.ndim == 4:
        dims = data_X.shape[1] * data_X.shape[2] * data_X.shape[3]

    while True:
        indices = np.arange(data_X.shape[0])
        np.random.shuffle(indices)
        for i in range(data_X.shape[0]//batch_size):
            current_indices = indices[i*batch_size:(i+1)*batch_size]
            X_batch = (data_X[current_indices] / 255.0).reshape(-1, dims)
            # X to P
            P = TSNE.calculate_P(X_batch)
            yield X_batch, P

def train():
    (X_train, y_train), (_, _) = mnist.load_data()

    model = create_model()
    model.compile("adam", loss=TSNE.KLdivergence)

    cb = Sampling(model, X_train[:1000], y_train[:1000])
    model.fit_generator(generator(X_train), steps_per_epoch=X_train.shape[0]//5000,
                        epochs=20, callbacks=[cb])

    files = glob.glob("plots/*")
    with zipfile.ZipFile("plots.zip", "w") as zip:
        for f in files:
            zip.write(f)


if __name__ == "__main__":
    train()
