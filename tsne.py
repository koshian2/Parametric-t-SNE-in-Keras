import numpy as np
import tensorflow as tf
import keras.backend as K
import multiprocessing as mp

class TSNE:
    @classmethod
    def Hbeta(cls, D, beta):
        P = np.exp(-D * beta)
        sumP = np.sum(P) + 1e-14
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    @classmethod
    def x2p_job(cls, data):
        i, Di, tol, logU = data
        beta = 1.0
        betamin = -np.inf
        betamax = np.inf
        H, thisP = cls.Hbeta(Di, beta)

        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta
                if betamax == -np.inf:
                    beta = beta * 2
                else:
                    beta = (betamin + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta = beta / 2
                else:
                    beta = (betamin + betamax) / 2

            H, thisP = cls.Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1

        return i, thisP


    @classmethod
    def x2p(cls, X, perplexity=30.0):
        tol = 1e-5
        n = X.shape[0]
        logU = np.log(perplexity)

        sum_X = np.sum(np.square(X), axis=1)
        D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))

        idx = (1 - np.eye(n)).astype(bool)
        D = D[idx].reshape([n, -1])

        def generator():
            for i in range(n):
                yield i, D[i], tol, logU

        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.map(cls.x2p_job, generator())
        P = np.zeros([n, n])
        for i, thisP in result:
            P[i, idx[i]] = thisP

        return P

    @classmethod
    def calculate_P(cls, X_batch):
        P_batch = cls.x2p(X_batch)
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        P_batch = P_batch / (P_batch.sum()+1e-14)
        P_batch = np.maximum(P_batch, 1e-12)
        return P_batch

    @classmethod
    def KLdivergence(cls, P, Y):
        alpha = K.int_shape(Y)[1] - 1.
        sum_Y = K.sum(K.square(Y), axis=1)
        eps = K.variable(10e-15)
        D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
        Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
        eye_dim = tf.cast(tf.shape(Q)[0], tf.int32) # Keras-APIだとバッチサイズが取れないのでTFの関数を使う
        Q = Q * (1-tf.linalg.eye(eye_dim)) # 対角要素を0にする
        Q /= K.sum(Q)
        Q = K.maximum(Q, eps)
        C = K.log((P + eps) / (Q + eps))
        C = K.sum(P * C)
        return C
