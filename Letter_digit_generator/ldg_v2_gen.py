from keras.layers import Lambda, Input, Dense, Flatten, Conv2D, Conv2DTranspose
from keras.layers import Activation, BatchNormalization, Reshape, Concatenate
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
from keras import initializers
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

label_str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
lebel_dict = {i: v for i, v in enumerate(label_str)}
n_label = len(label_str)

img_rows = 28
img_cols = 28
img_channel = 1
orig_dimension = img_rows * img_cols
image_shape = (img_rows, img_cols, img_channel)
latent_dim = 10


def sampling(arg):
    arg = [z_mean, z_log_var]
    dim = K.int_shape(z_mean)[1]
    # reparameterization trick
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


w_init = initializers.random_normal(stddev=0.02)
gamma_init = initializers.random_normal(mean=1.0, stddev=0.02)

# encoder
img_inputs = Input(shape=(orig_dimension,), name='image_input')
label_inputs = Input(shape=(n_label,), name='label_input')
encoder_inputs = Concatenate()([img_inputs, label_inputs])

x = Dense(orig_dimension, kernel_initializer=w_init,
          activation='relu')(encoder_inputs)

x = Reshape(image_shape)(x)
x = Conv2D(16, 3, strides=1, padding='same', kernel_initializer=w_init)(x)
x = BatchNormalization(gamma_initializer=gamma_init)(x)
x = Activation('relu')(x)
x = Conv2D(32, 3, strides=1, padding='same', kernel_initializer=w_init)(x)
x = BatchNormalization(gamma_initializer=gamma_init)(x)
x = Activation('relu')(x)
x = Conv2D(64, 3, strides=2, padding='same', kernel_initializer=w_init)(x)
x = BatchNormalization(gamma_initializer=gamma_init)(x)
x = Activation('relu')(x)
before_flatten_shape = K.int_shape(x)
x = Flatten()(x)
x = Dense(128, kernel_initializer=w_init, activation='relu')(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model([img_inputs, label_inputs], [
                z_mean, z_log_var, z], name='encoder')

# decoder
latent_inputs = Input(shape=(latent_dim,), name='latent_inputs')
decoder_inputs = Concatenate()([latent_inputs, label_inputs])

x = Dense(128, kernel_initializer=w_init, activation='relu')(decoder_inputs)
x = Dense(before_flatten_shape[1] * before_flatten_shape[2] *
          before_flatten_shape[3], activation='relu', kernel_initializer=w_init)(x)
x = Reshape(
    (before_flatten_shape[1], before_flatten_shape[2], before_flatten_shape[3]))(x)

x = Conv2DTranspose(64, 3, strides=1, padding='same',
                    kernel_initializer=w_init)(x)
x = BatchNormalization(gamma_initializer=gamma_init)(x)
x = Activation('relu')(x)

x = Conv2DTranspose(32, 3, strides=2, padding='same',
                    kernel_initializer=w_init)(x)
x = BatchNormalization(gamma_initializer=gamma_init)(x)
x = Activation('relu')(x)

x = Conv2DTranspose(16, 3, strides=1, padding='same',
                    kernel_initializer=w_init)(x)
x = BatchNormalization(gamma_initializer=gamma_init)(x)
x = Activation('relu')(x)

x = Conv2DTranspose(img_channel, 3, activation='tanh',
                    padding='same', kernel_initializer=w_init)(x)
outputs = Flatten()(x)

# instantiate decoder model
decoder = Model([latent_inputs, label_inputs], outputs, name='decoder')

# VAE
outputs = decoder([encoder([img_inputs, label_inputs])[2], label_inputs])
vae = Model([img_inputs, label_inputs], outputs)


def load_weights_vae(weight_name=None):
    # load all the weights for encoder and decoder when loading for vae
    vae.load_weights(os.path.join(saving_folder, weight_name))


def find_idx_from_label_dict(search_str):
    for i, val in lebel_dict.items():
        if val == search_str:
            return i


def letter_digit_gen(input_str, th=None):
    gap = 20
    img = np.zeros((img_cols, gap * len(input_str) + (img_cols - gap)))
    for idx, l in enumerate(input_str):
        if l == ' ':
            img[:, gap * idx:gap * idx +
                img_cols] += np.zeros((img_cols, img_cols))
        elif l not in label_str:
            pass
        else:
            cls_idx = find_idx_from_label_dict(l)
            latent = np.random.randn(latent_dim)
            latent = np.expand_dims(np.random.randn(latent_dim), 0)
            generated = decoder.predict(
                [latent, np.expand_dims(to_categorical(cls_idx, n_label), 0)])
            generated = (0.5 * generated) + 0.5
            generated = generated.reshape(img_rows, img_cols)
            generated = np.transpose(generated)
            img[:, gap * idx:gap * idx + img_cols] += generated
    if th != None:
        for i, v in np.ndenumerate(img):
            if v >= th:
                img[i] = 1
            else:
                img[i] = 0
    return img


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(
            '{} is not in range [0.0, 1.0]'.format(x))
    return x


parser = argparse.ArgumentParser()
parser.add_argument('input_str', help='input string to be converted')
parser.add_argument('-t', '--threshold', type=restricted_float,
                    help='binary threshold: float[0-1]')
parser.add_argument('-s', '--save',
                    help='save the image in .png')
args = parser.parse_args()

input_str = args.input_str

saving_folder = 'best_weight_ldg_v2_conv-cvae'
load_weights_vae(
    weight_name='ldg_v2_conv-cvae-best-wiehgts-050-131.521-131.939.h5')

img = letter_digit_gen(input_str, th=args.threshold)
plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(1 - img, cmap='gray')

if args.save:
    os.makedirs('generated_images', exist_ok=True)
    plt.savefig('generated_images/{}.png'.format(args.save))

plt.show()
