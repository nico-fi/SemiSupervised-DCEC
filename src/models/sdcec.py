import numpy as np
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Layer, InputSpec, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.utils import to_categorical


class ClusteringLayer(Layer):
    """
    Converts input sample to soft label, i.e. a vector representing the probability of the
    sample to belong to each cluster. The probability is calculated with Student's t-distribution.
    Arguments:
        n_clusters: number of clusters
        alpha: parameter in Student's t-distribution. Default to 1.0
    Input shape: (n_samples, n_features)
    Output shape: (n_samples, n_clusters)
    """
    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_shape[1]))
        self.clusters = self.add_weight('clusters', (self.n_clusters, input_shape[1]), initializer='glorot_uniform')

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q


class SDCEC:
    
    def __init__(self, input_shape, n_clusters):
        self.n_clusters = n_clusters
        filters = [32, 64, 128, 10]
        pad = 'same' if input_shape[0] % 8 == 0 else 'valid'
        self.autoencoder = Sequential([
            Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape),
            Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'),
            Conv2D(filters[2], 3, strides=2, padding=pad, activation='relu', name='conv3'),
            Flatten(),
            Dense(units=filters[3], name='embedding'),
            Dense(units=filters[2] * (input_shape[0] // 8) ** 2, activation='relu'),
            Reshape((input_shape[0] // 8, input_shape[0] // 8, filters[2])),
            Conv2DTranspose(filters[1], 3, strides=2, padding=pad, activation='relu', name='deconv3'),
            Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'),
            Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')
        ])
        embedding = self.autoencoder.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=embedding)
        clustering = ClusteringLayer(self.n_clusters, name='clustering')(embedding)
        self.model = Model(inputs=self.autoencoder.input, outputs=clustering)

    def compile(self):
        self.model.compile(loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam')

    def fit(self, X, y_train, batch_size, epochs, max_iter, tol):
        print('Pretraining autoencoder...')
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs)

        print('Initializing cluster centers...')
        encodings = self.encoder.predict(X)
        cluster_centers = [np.mean([e for (e, y) in zip(encodings, y_train) if y == n], axis = 0) for n in range(self.n_clusters)]
        self.model.get_layer(name='clustering').set_weights([np.array(cluster_centers)])
        known_instances = np.where(y_train >= 0)
        p_fixed = to_categorical(y_train[known_instances])

        print('Deep clustering...')
        start = 0
        update_interval = 140
        for iter in range(max_iter):
            if iter % update_interval == 0:
                q = self.model.predict(X, verbose=0)
                p = q ** 2 / q.sum(0)
                p = (p.T / p.sum(1)).T
                p[known_instances] = p_fixed
                
                predictions = q.argmax(1)
                if iter > 0:
                    delta = np.sum(predictions != last_predictions) / predictions.shape[0]
                    if delta < tol:
                        print('Reached tolerance threshold')
                        break
                    print(f'Iter {iter}: loss={loss}')
                last_predictions = np.copy(predictions)

            end = start + batch_size
            loss = self.model.train_on_batch(x=X[start:end], y=[p[start:end], X[start:end]])
            start = end if end < len(X) else 0

    def save(self, path):
        self.model.save(path)