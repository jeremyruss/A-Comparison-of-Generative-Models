import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import math
from loaddata import load_data

class CVAE(tf.keras.Model):

	def __init__(self, latent_dim):
		super(CVAE, self).__init__()
		self.latent_dim = latent_dim
		self.encoder = tf.keras.Sequential(
				[		
					tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
					tf.keras.layers.Conv2D(
							filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
					tf.keras.layers.Conv2D(
							filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
					tf.keras.layers.Flatten(),
					tf.keras.layers.Dense(latent_dim + latent_dim),
				]
		)

		self.decoder = tf.keras.Sequential(
				[
					tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
					tf.keras.layers.Dense(
						units=8*8*32, activation=tf.nn.relu),
					tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
					tf.keras.layers.Conv2DTranspose(
						filters=64, kernel_size=3, strides=2, padding='same',
						activation='relu'),
					tf.keras.layers.Conv2DTranspose(
						filters=32, kernel_size=3, strides=2, padding='same',
						activation='relu'),
					tf.keras.layers.Conv2DTranspose(
						filters=3, kernel_size=3, strides=1, padding='same'),
				]
		)

	@tf.function
	def sample(self, eps=None):
		if eps is None:
			eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	def encode(self, x):
		mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		eps = tf.random.normal(shape=mean.shape)
		return eps * tf.exp(logvar * .5) + mean

	def decode(self, z, apply_sigmoid=False):
		logits = self.decoder(z)
		if apply_sigmoid:
			probs = tf.sigmoid(logits)
			return probs
		return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
	log2pi = tf.math.log(2. * np.pi)
	return tf.reduce_sum(
			-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
			axis=raxis)


def compute_loss(model, x):
	mean, logvar = model.encode(x)
	z = model.reparameterize(mean, logvar)
	x_logit = model.decode(z)
	x_logit = tf.cast(x_logit, 'float32')
	x = tf.cast(x, 'float32')
	cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
	logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
	logpz = log_normal_pdf(z, 0., 0.)
	logqz_x = log_normal_pdf(z, mean, logvar)
	return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
	with tf.GradientTape() as tape:
		loss = compute_loss(model, x)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))


#Begin training

optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 999999
LATENT_DIM = 2
NUM_EXAMPLES = 64
BATCH_SIZE = 128
TRAIN_TIME = 43200

x_train, _ = load_data()
x_train, x_test = train_test_split(x_train, test_size=0.1, random_state=42)

train_size = x_train.shape[0]
test_size = x_test.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices(
		x_train).shuffle(train_size).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
		x_test).shuffle(test_size).batch(BATCH_SIZE)

random_vector = tf.random.normal(shape=[NUM_EXAMPLES, LATENT_DIM])
model = CVAE(LATENT_DIM)

assert BATCH_SIZE >= NUM_EXAMPLES

def train(train_dataset, test_dataset, epochs, batch_size, model):
	train_start = time.time()
	for epoch in range(epochs):
		start = time.time()
		for train_batch in train_dataset:
			train_step(model, train_batch, optimizer)
		end = time.time()
		train_time = end - train_start
		loss = tf.keras.metrics.Mean()
		for test_batch in test_dataset:
			loss(compute_loss(model, test_batch))
		elbo = -loss.result()
		print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end - start))
		for test_batch in test_dataset.take(1):
			test_sample = test_batch[0:NUM_EXAMPLES, :, :, :]
		if epoch % 120 == 0:
			generate_images(model, epoch, test_sample)
		if train_time > TRAIN_TIME:
			generate_images(model, epoch, test_sample)
			print('Finished!')
			sys.exit()

def generate_images(model, epoch, test_input):
	n = math.sqrt(NUM_EXAMPLES)
	mean, logvar = model.encode(test_input)
	z = model.reparameterize(mean, logvar)
	predictions = model.sample(z)
	fig = plt.figure(figsize=(n,n))
	for i in range(predictions.shape[0]):
		plt.subplot(n, n, i + 1)
		plt.imshow(predictions[i])
		plt.axis('off')
	plt.subplots_adjust(wspace=0.05, hspace=0.05)
	plt.savefig('./output/cvae/image_at_epoch_{:04d}.png'.format(epoch), bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
	train(train_dataset, test_dataset, EPOCHS, BATCH_SIZE, model)
