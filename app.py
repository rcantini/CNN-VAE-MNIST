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
# create model
vae, models = CVAE.create_compiled_model()
(encoder, decoder) = models
# load weights
vae.load_weights('save/cvae_mnist.h5')
# fixed means and log_vars from latent space inspection
means = [[ 0.0737659,   2.08392119],
 [ 1.02552891, -0.96927363],
 [-0.24872708,  0.32876885],
 [ 0.20523334,  0.1215527 ],
 [-3.2,  0.65],
 [ 2.1,  0.55],
 [-0.19239137,  0.83706224],
 [-1.3523798,  -1.29118562],
 [ 0.59725612, -0.16651063],
 [-0.5174467,  -0.50842798]]
log_vars = [[-6.5435915,  -5.83439541],
 [-6.61758423, -6.32054329],
 [-6.985816,   -8.72492027],
 [-7.50982904, -8.93076611],
 [-9.70335436, -9.74416351],
 [-6.42503643, -8.40911388],
 [-6.1917448,  -7.79825306],
 [-5.25109243, -5.8355608 ],
 [-6.42724133, -7.7093935 ],
 [-5.72645664, -6.8247509 ]]
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
    app.run()
