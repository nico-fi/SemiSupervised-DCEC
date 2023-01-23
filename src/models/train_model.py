"""
This script trains the model and saves it.
"""

from pathlib import Path
import random
import numpy as np
import yaml
import mlflow
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Layer, InputSpec, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.utils import to_categorical
from tensorflow import random as tf_random


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
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_spec = InputSpec(ndim=2)
        self.clusters = None

    def build(self, input_shape):
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_shape[1]))
        self.clusters = self.add_weight(
            'clusters',
            (self.n_clusters, input_shape[1]),
            initializer='glorot_uniform'
        )

    def call(self, inputs, *args, **kwargs):
        q_dist = 1.0 / (1.0 + (K.sum(K.square(
            K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q_dist **= (self.alpha + 1.0) / 2.0
        q_dist = K.transpose(K.transpose(q_dist) / K.sum(q_dist, axis=1))
        return q_dist


class SDCEC:
    """
    Perform semi-supervised deep clustering with convolutional autoencoder.
    Arguments:
        input_shape: shape of input data
        n_clusters: number of clusters
    """
    def __init__(self, input_shape, n_clusters):
        self.n_clusters = n_clusters
        filters = [32, 64, 128, 10]
        pad = 'same' if input_shape[0] % 8 == 0 else 'valid'
        self.autoencoder = Sequential([
            Conv2D(filters[0], 5, 2, 'same', activation='relu', name='conv1',
                input_shape=input_shape),
            Conv2D(filters[1], 5, 2, 'same', activation='relu', name='conv2'),
            Conv2D(filters[2], 3, 2, pad, activation='relu', name='conv3'),
            Flatten(),
            Dense(units=filters[3], name='embedding'),
            Dense(units=filters[2] * (input_shape[0] // 8) ** 2, activation='relu'),
            Reshape((input_shape[0] // 8, input_shape[0] // 8, filters[2])),
            Conv2DTranspose(filters[1], 3, 2, pad, activation='relu', name='deconv3'),
            Conv2DTranspose(filters[0], 5, 2, 'same', activation='relu', name='deconv2'),
            Conv2DTranspose(input_shape[2], 5, 2, 'same', name='deconv1')
        ])
        embedding = self.autoencoder.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=embedding)
        clustering = ClusteringLayer(self.n_clusters, name='clustering')(embedding)
        self.model = Model(inputs=self.autoencoder.input, outputs=clustering)
        self.model.compile(loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam')


    def fit(self, x_data, y_data, batch_size, epochs, max_iter, tol):
        """
        Trains the model.
        Arguments:
            x_data: input data
            y_data: data labels. If label is -1, the sample is considered unlabeled
            batch_size: size of the batch
            epochs: number of epochs for autoencoder pretraining
            max_iter: maximum number of iterations for clustering
            tol: tolerance threshold for stopping clustering
        """
        print('Pretraining autoencoder...')
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(x_data, x_data, batch_size=batch_size, epochs=epochs)

        print('Initializing cluster centers...')
        encodings = self.encoder.predict(x_data)
        cluster_centers = [np.mean([e for (e, y) in zip(encodings, y_data) if y == n], axis = 0)
            for n in range(self.n_clusters)]
        self.model.get_layer(name='clustering').set_weights([np.array(cluster_centers)])
        known_instances = np.where(y_data >= 0)
        p_fixed = to_categorical(y_data[known_instances])

        print('Deep clustering...')
        start = 0
        update_interval = 140
        for ite in range(max_iter):
            if ite % update_interval == 0:
                q_dist = self.model.predict(x_data, verbose=0)
                p_dist = q_dist ** 2 / q_dist.sum(0)
                p_dist = (p_dist.T / p_dist.sum(1)).T
                p_dist[known_instances] = p_fixed
                predictions = q_dist.argmax(1)
                if ite > 0 and np.sum(predictions != last_predictions) / predictions.shape[0] < tol:
                    print('Reached tolerance threshold')
                    break
                print(f'Iter {ite}')
                last_predictions = np.copy(predictions)
            end = start + batch_size
            self.model.train_on_batch(x_data[start:end], [p_dist[start:end], x_data[start:end]])
            start = end if end < len(x_data) else 0


    def save(self, path):
        """
        Saves the model.
        Arguments:
            path: path to save the model
        """
        self.model.save(path)


def main():
    """
    Trains the model.
    """
    mlflow.set_tracking_uri("https://dagshub.com/nico-fi/SemiSupervised-DCEC.mlflow")
    mlflow.set_experiment("Train Model")
    mlflow.start_run()

    params_path = Path("params.yaml")
    input_folder_path = Path("data/processed")
    output_folder_path = Path("models")

    # Read training data
    x_data = np.load(input_folder_path / "x.npy")
    y_train = np.load(input_folder_path / "y_train.npy")

    # Load and log training parameters
    with open(params_path, "r", encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)["train"]
    mlflow.log_params(
        {
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "max_iter": params["max_iter"],
            "tol": params["tol"],
            "random_state": params["random_state"],
        }
    )

    # Set seeds
    np.random.seed(params["random_state"])
    random.seed(params["random_state"])
    tf_random.set_seed(params["random_state"])

    # Train the model
    model = SDCEC(input_shape=x_data.shape[1:], n_clusters=len(np.unique(y_train)) - 1)
    model.fit(
        x_data,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        max_iter=params["max_iter"],
        tol=params["tol"]
    )

    # Save and log the model
    output_folder_path.mkdir(exist_ok=True)
    model_file_path = output_folder_path / "model.tf"
    model.save(model_file_path)
    mlflow.log_artifact(model_file_path)

    mlflow.end_run()


if __name__ == "__main__": # pragma: no cover
    main()
