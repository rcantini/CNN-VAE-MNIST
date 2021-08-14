import numpy as np
from keras import backend as K
import CVAE
from flask import Flask, render_template, request

# network parameters
batch_size = 128
latent_dim = 2
epochs = 80
image_shape = (28, 28, 1)
original_dim = image_shape[0] * image_shape[1]

# load MNIST data
x_train, y_train = CVAE.load_data()
vae, models = CVAE.create_compiled_model(x_train)
(encoder, decoder) = models
# load weights
vae.load_weights('save/cvae_mnist.h5')
# fixed means and log_vars
# -- latent space inspection
z_mean, z_log_var, _ = encoder.predict(x_train, batch_size=batch_size)
z0_mean = z_mean[:, 0]
z1_mean = z_mean[:, 1]
z0_logvar = z_log_var[:, 0]
z1_logvar = z_log_var[:, 1]
means = np.zeros((10, 2))
log_vars = np.zeros((10, 2))
for i in range(0, 10):
    indexes = y_train == i
    means[i] = [np.median(z0_mean[indexes]), np.median(z1_mean[indexes])]
    log_vars[i] = [np.median(z0_logvar[indexes]), np.median(z1_logvar[indexes])]
# -- improvement
means[4] = [-3.2, 0.65]
log_vars[4] = [-10, -10]
means[5] = [2.1, 0.55]
# init app
print(__name__)
app = Flask(__name__)


def draw_digit(z_mean, z_log_var):
    random_normal = K.random_normal(shape=(1, 2), dtype=np.float64)
    eps = K.get_value(random_normal)
    z_sampled = z_mean + np.array(K.get_value(K.exp(0.5 * np.array(z_log_var)))) * eps
    drawn = decoder.predict(z_sampled)
    drawn = drawn[0].tolist()
    return drawn


def guess(z0, z1):
    z_log_var = [-10, -10]
    drawn = draw_digit([z0, z1], z_log_var)
    return drawn


def draw(n):
    drawn = np.zeros((1, 28*28)).tolist()
    if 0 <= n <= 9:
        z_mean = means[n]
        z_log_var = log_vars[n]
        drawn = draw_digit(z_mean, z_log_var)
    return drawn


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def handle():
    drawn = np.zeros((1, 28 * 28)).tolist()
    if request.method == 'POST':
        try:
            if 'z0' in request.form.keys() and 'z1' in request.form.keys():
                z0 = float(request.form['z0'])
                z1 = float(request.form['z1'])
                drawn = guess(z0, z1)
            elif 'n' in request.form.keys():
                n = int(request.form['n'])
                drawn = draw(n)
        except:
            pass
    return render_template('home.html', result=drawn)


if __name__ == '__main__':
    app.run(debug=True)
