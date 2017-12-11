import tensorflow as tf
import librosa
import os
from IPython.display import display, Audio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow.contrib import rnn

# Defining parameters
CONTENT_FILENAME = "neural-style-audio-tf/inputs/bach_violin.mp3"
STYLE_FILENAME = "neural-style-audio-tf/inputs/elgar_cello.mp3"

# Reads wav file and produces spectrum
# Fourier phases are ignored
N_FFT = 2048
def read_audio_spectrum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)

    S = np.log1p(np.abs(S[:,:430]))
    return S, fs

# get the spectograms for the content and style references
a_content, fs = read_audio_spectrum(CONTENT_FILENAME)
a_style, fs = read_audio_spectrum(STYLE_FILENAME)

N_CHANNELS = a_content.shape[0]
N_SAMPLES = a_content.shape[1]
a_style = a_style[:N_CHANNELS, :N_SAMPLES] # re-size

num_features = N_CHANNELS # size of each input vector
time_steps = N_SAMPLES # number of input vectors to expect
num_hidden_units = 1025
batch_size = 1
learning_rate = .001


# Input arrays
a_content_tf = np.ascontiguousarray(a_content.T[None,:,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,:,:])


# Building LSTM
g = tf.Graph()

with g.as_default(), g.device('/cpu:0'):

    # Randomize initial state of rnn
    x = tf.placeholder("float32", [1, time_steps, num_features], name="x")
    inp = tf.unstack(x, time_steps, 1) # input to rnn

    lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_hidden_units) #can add forget_bias=1 as arg
    init_state = lstm_layer.zero_state(batch_size, tf.float32)
    outputs, state = tf.contrib.rnn.static_rnn(lstm_layer, inp, initial_state=init_state, dtype="float32")
    out = tf.concat(outputs, 0)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        content_features = sess.run(out, feed_dict={x:a_content_tf})
        style_features = sess.run(out, feed_dict={x:a_style_tf})

        style_gram = np.matmul(style_features.T, style_features) / N_SAMPLES  # TODO maybe need to resize this?


# Running output spectrogram through LSTM
from sys import stderr

ALPHA= 1e-2
learning_rate= 1e-3
iterations = 100

result = None
with tf.Graph().as_default():

    # Build graph with variable input
    x = tf.Variable(np.random.randn(1,time_steps, num_features).astype(np.float32)*1e-3, name="x")
    inp = tf.unstack(x, time_steps, 1) # input to rnn

    lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_hidden_units)
    init_state = lstm_layer.zero_state(batch_size, tf.float32)
    outputs, state = tf.contrib.rnn.static_rnn(lstm_layer, inp, initial_state=init_state, dtype="float32")
    out = tf.concat(outputs, 0)


    content_loss = ALPHA * 2 * tf.nn.l2_loss(
            out - content_features)

    gram = tf.matmul(tf.transpose(out), out)  / N_SAMPLES
    style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

     # Overall loss
    loss = content_loss + style_loss

    opt = tf.contrib.opt.ScipyOptimizerInterface(
          loss, method='L-BFGS-B', options={'maxiter': 300})

    # Optimization
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        print('Started optimization.')
        opt.minimize(sess)

        print 'Final loss:', loss.eval()
        result = x.eval()

a = np.zeros_like(a_content)
a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

# This code is supposed to do phase reconstruction
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, N_FFT))

OUTPUT_FILENAME = 'outputs/' + 'rnn.wav' # TODO: Change this filename to not overwrite results
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)

print OUTPUT_FILENAME
display(Audio(OUTPUT_FILENAME))
