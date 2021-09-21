# CNN-VAE-MNIST

This repository is dedicated to the development of a **Flask** web application capable of drawing digits through the use of a generative model. This model is obtained by training a convolutional variational autoencoder on the MNIST dataset of handwritten digits using **Keras**+**Tensorflow**.

The web application, named ***Let's play with MNIST***, was deployed on Heroku and is available at this link: https://play-with-mnist.herokuapp.com/

I also embedded it directly here so that you can try it now!
In particular you can use it in two ways:
- Choose a digit (i.e. a number from \\( 0 \\) to \\( 9 \\)) to be drawn by the VAE.
- Explore the 2D latent space by inserting the coordinates of a random point and see which digit is generated.

Have fun! :laughing::laughing:

<iframe src="https://play-with-mnist.herokuapp.com/" width="100%" height="708px"></iframe>

For further information, check out my blog post: https://riccardo-cantini.netlify.app/post/bert_text_classification/
