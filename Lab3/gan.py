from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.layers import Dense, LeakyReLU, ReLU, Layer, Input, Conv2D, BatchNormalization,Conv2DTranspose, Flatten, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.network import Network
import tensorflow as tf

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import glob
import imageio
import matplotlib.pyplot as plt
#from matplotlib.pyplot import ion
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

tf.enable_eager_execution()
#ion()
# Load and prepare the dataset
# You will use the MNIST dataset to train the generator and the discriminator. The generator will generate handwritten digits resembling the MNIST data.

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 1

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Create the models

# Both the generator and discriminator are defined using the Keras Sequential API.
# The Generator

# The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). Start with a Dense layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. Notice the tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh.

def make_generator_model(k,name=None):
    # k = 3
    in_norm = InstanceNormalization()

    input = Input(shape=[100,])
    x = Dense(7*7*32*2**k, use_bias=False)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Reshape((7, 7, 32*2**k))(x)

    x = Conv2DTranspose(16*2**k, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x) #in_norm(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(8*2**k, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x) #in_norm(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

    return Model(inputs=input, outputs=x, name=name)

# Use the (as yet untrained) generator to create an image.

generator = make_generator_model(3)
print(generator.summary())

# The Discriminator

def make_discriminator_model(name=None):

    input = Input(shape=[28, 28, 1])
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same',)(input)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    return Model(inputs=input, outputs=x, name=name)

# Use the (as yet untrained) discriminator to classify the generated images as real or fake. The model will be trained to output positive values for real images, and negative values for fake images.

discriminator = make_discriminator_model()

# Build discriminator

input_data = Input(shape=[28,28,1])
guess = discriminator(input_data)
D = Model(inputs=input_data, outputs=guess, name='D_model')
D.compile(optimizer=tf.train.AdamOptimizer(1e-4),loss=tf.keras.losses.binary_crossentropy)

# Build generator

D_static = Network(inputs=input_data,outputs=guess,name='D_static_model')
D_static.trainable = False
noise_vector = Input(shape=[100,],name='real_A')
synthetic_data = generator(noise_vector)
guess_synthetic = D_static(synthetic_data)
G = Model(inputs=noise_vector,outputs=guess_synthetic,name='G_model')
G.compile(optimizer=tf.train.AdamOptimizer(1e-4),loss=tf.keras.losses.binary_crossentropy)

# Define the training loop

EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random_normal([num_examples_to_generate, noise_dim])

# The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.

def train_step(images):
    SIZE = np.shape(images)[0]
    noise = tf.random_normal([SIZE, noise_dim])

    G_loss = G.train_on_batch(x=noise,y=tf.constant(np.ones((SIZE,1))))
    generated_images = generator.predict(noise,batch_size=int(SIZE))
    D_loss_true = D.train_on_batch(x=images,y=tf.constant(np.ones((SIZE,1))))
    D_loss_false = D.train_on_batch(x=tf.constant(generated_images),y=tf.constant(np.zeros((SIZE,1))))
    D_loss = D_loss_true + D_loss_false
    return G_loss, D_loss

def train(dataset, epochs):
    losses = []
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            G_loss,D_loss = train_step(image_batch)
            losses.append([G_loss,D_loss])
   
        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

    #    # Save the model every 15 epochs
    #    if (epoch + 1) % 15 == 0:
    #      checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, np.round(time.time()-start)))
        
    # Generate after the final epoch
    display.clear_output(wait=False)
    generate_and_save_images(generator,
                            epochs,
                            seed)
    return losses

# Generate and save images

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model.predict(test_input)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        #plt.title('Epoch '+str(epoch))

    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    #if epoch ==1 or epoch % 10 == 0:
    #    plt.show()

# Train the model

# Call the train() method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).

# At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.

#%%time
losses = train(train_dataset, EPOCHS)
plt.show()
np.save('images/losses.npy',losses)
