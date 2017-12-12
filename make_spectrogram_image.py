import tensorflow as tf
import librosa
import os
from IPython.display import display, Audio
import numpy as np
import matplotlib.pyplot as plt
import argparse

N_FFT = 2048
CONTENT_FILENAME = "audio_files/clarinet.mp3"
STYLE_FILENAME = "audio_files/piano.mp3"
THIRD_FILENAME = "audio_files/violin.wav"
fig1_name = "base_spectrograms.png"

def read_audio_spectrum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs


def main():
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-c','--content', help='Path to content filename', required=True)
	parser.add_argument('-s','--style', help='Path to style filename', required=True)
	parser.add_argument('-f', '--final', help='path to final combination audio', required=True)
	parser.add_argument('-o', '--output', help='Name for output path', required=True)
	args = vars(parser.parse_args())
	CONTENT_FILENAME = args['content']
	STYLE_FILENAME = args['style']
	THIRD_FILENAME = args['final']
	fig1_name = args['output']

	# get the spectograms for the content and style references
	a1, fs = read_audio_spectrum(CONTENT_FILENAME)
	a2, fs = read_audio_spectrum(STYLE_FILENAME)
	a3, fs = read_audio_spectrum(THIRD_FILENAME)	
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 3, 1)
	plt.title('Clarinet')
	plt.imshow(a1[:,:])
	plt.subplot(1, 3, 2)
	plt.title('Piano')
	plt.imshow(a2[:,:])
	plt.subplot(1, 3, 3)
	plt.title('Violin')
	plt.imshow(a3[:,:])
	
	plt.savefig(fig1_name)
	plt.show()


if __name__ == '__main__':
	main()