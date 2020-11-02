import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import math
import sys
from loaddata import load_data

def generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Reshape((8, 8, 256)))
	assert model.output_shape == (None, 8, 8, 256)

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 8, 8, 128)

	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 16, 16, 64)

	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 32, 32, 3)
	return model


def discriminator_model():
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))
	return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

generator = generator_model()
discriminator = discriminator_model()

generator_opt = tf.keras.optimizers.Adam(1e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt,
								 discriminator_optimizer=discriminator_opt,
								 generator=generator,
								 discriminator=discriminator)

ckpdir = './checkpoints/'
ckpprefix = os.path.join(ckpdir, "ckp")

x_train, _ = load_data()

EPOCHS = 999999
BATCH_SIZE = 32
BUFFER_SIZE = len(x_train)
TRAIN_TIME = 43200

noise_dim = 100
num_examples = 64

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

@tf.function
def train_step(images):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
	  generated_images = generator(noise, training=True)

	  real_output = discriminator(images, training=True)
	  fake_output = discriminator(generated_images, training=True)

	  gen_loss = generator_loss(fake_output)
	  disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, batch_size):
	train_start = time.time()
	for epoch in range(epochs):
		start = time.time()
		for batch in dataset:
			train_step(batch)
		end = time.time()
		train_time = end - train_start
		print('Time for epoch {} is {} sec'.format(epoch, train_time))
		seed = tf.random.normal([num_examples, noise_dim])
		if epoch % 9 == 0:
			generate_images(generator, epoch, seed)
		if train_time > TRAIN_TIME:
			generate_images(generator, epoch, seed)
			print('Finished!')
			sys.exit()

def generate_images(model, epoch, test_input):
	n = math.sqrt(num_examples)
	predictions = model(test_input, training=False)
	predictions = (predictions + 1) / 2
	fig = plt.figure(figsize=(n, n))
	for i in range(predictions.shape[0]):
		plt.subplot(n, n, i+1)
		plt.imshow(predictions[i])
		plt.axis('off')
	plt.subplots_adjust(wspace=0.05, hspace=0.05)
	plt.savefig('./output/dcgan/image_at_epoch_{:04d}.png'.format(epoch), bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
	train(train_dataset, EPOCHS, BATCH_SIZE)
