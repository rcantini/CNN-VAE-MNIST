import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras import layers
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.utils.vis_utils import plot_model


# network parameters
batch_size = 128
latent_dim = 2
epochs = 80
image_shape = (28, 28, 1)
original_dim = image_shape[0] * image_shape[1]


def load_data():
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.row_stack((x_train, x_test))
    y_train = np.append(y_train, y_test)
    x_train = np.reshape(x_train, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    return x_train, y_train


### reparameterization trick ###
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def crate_compiled_model(x_train):
    os.makedirs("plots", exist_ok=True)
    os.makedirs("save", exist_ok=True)
    # VAE model = encoder + decoder
    # build encoder model
    input_shape = (original_dim,)
    inputs = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Reshape(image_shape)(inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick
    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='plots/encoder.png', show_shapes=True)
    # build decoder model
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(7 * 7 * 64, activation='relu')(x)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(x)
    outputs = layers.Reshape(input_shape)(outputs)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='plots/decoder.png', show_shapes=True)
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='CVAE')
    vae.summary()
    plot_model(vae, to_file='plots/CVAE.png', show_shapes=True)
    models = (encoder, decoder)
    # VAE loss = xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, models


def plot_results(models, data, batch_size=128):
    encoder, decoder = models
    x, y = data
    filename = os.path.join("plots", "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()
    filename = os.path.join("plots", "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 40
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size*2)
    sample_range_x = np.round(np.linspace(-4, 4, int(n/2)), 1)
    sample_range_y = np.round(np.linspace(-4, 4, int(n/2))[::-1], 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    # load MNIST data
    x_train, y_train = load_data()
    data = (x_train, y_train)
    # train the autoencoder
    vae, models = crate_compiled_model(x_train)
    vae.fit(x_train,x_train,
            epochs=epochs,
            batch_size=batch_size)
    # serialize model to JSON
    model_json = vae.to_json()
    with open("save/model.json", "w") as json_file:
        json_file.write(model_json)
    vae.save_weights('save/cvae_mnist.h5')
    # plot labels and MNIST digits as function of 2-D latent vector
    plot_results(models, data, batch_size=batch_size)
