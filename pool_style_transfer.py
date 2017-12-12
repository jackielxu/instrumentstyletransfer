import tensorflow as tf
import librosa
import os
from IPython.display import display, Audio
import numpy as np
import matplotlib.pyplot as plt
import argparse
N_FFT = 2048
NUM_SAMPLES = 430
NUM_CHANNELS = 1025 # ends up being this way from this many dimension
STRIDES = [1, 1, 1, 1]
PADDING = "VALID"
LEARNING_RATE=1e-3
ITERATIONS = 5
ALPHA= 1e-2
CONTENT = "content"
RANDOM = "random"

# TODO: also try mel vs. not mel spectogram
# TODO: need to make it so the optimization doesn't use sess.run - separate function that does it through variables etc.
# input layer turns 1025 channels ->

def read_audio_spectrum(filename):
	x, fs = librosa.load(filename)
	S = librosa.stft(x, N_FFT)
	p = np.angle(S)
	S = np.log1p(np.abs(S[:,:NUM_CHANNELS]))
        S = S[:NUM_CHANNELS, :NUM_SAMPLES]
	return S, fs # spectogram and phase

def get_style_features(content_features, size, is_tensor):
    """
    Given the **content** output, returns the style output for that layer
    size - the size of the content features probably
    """
    if not is_tensor:
        ret = np.reshape(content_features, (-1, size))
        ret_gram = np.matmul(ret.T, ret)/ NUM_SAMPLES
    else:
        ret = tf.reshape(content_features, (-1, size))
        ret_gram = tf.matmul(tf.transpose(ret), ret)/ NUM_SAMPLES
    return ret_gram

def get_content_loss(content_features_a, content_features_b):
    return ALPHA * 2 * tf.nn.l2_loss(content_features_a - content_features_b)

def get_style_loss(gram1, gram2):
    """
    gram1 - new; gram2 - style reference
    """
    print ("shape1: ", np.shape(gram1))
    print ("shape2: ", np.shape(gram2))
    print ("val: ", tf.nn.l2_loss(gram1 - gram2))
    return 2 * tf.nn.l2_loss(gram1 - gram2)

def initialize_kernels(num_layers, num_filters, filter_width):
    """
    For model - initializes the kernels we will be using to build the layers of the network
    """
    ret_kernel = []
    layer1_std = np.sqrt(2) * np.sqrt(2.0/((NUM_CHANNELS + num_filters))*filter_width)
    layer1_kernel = np.random.randn(1, filter_width, NUM_CHANNELS, num_filters)*layer1_std
    ret_kernel.append(layer1_kernel)
    for i in range(1, num_layers):
        layer_filters = num_filters/(i+1)
        std = np.sqrt(2) * np.sqrt( 2.0 / ((num_filters + layer_filters) * filter_width))
	kernel = np.random.randn(1, filter_width, num_filters, layer_filters)*std
	ret_kernel.append(kernel)
    return ret_kernel

def run_model(x, tf_layers, input_vector, sess):
    output_vectors = sess.run(tf_layers, feed_dict = {x: input_vector})
    print len(output_vectors)
    return output_vectors

def run_optimization(random_kernels, content_refs, style_refs, num_filters, num_layers, filter_width, starting, content_input_tf):
    style_loss_ratios = [1.0/float(num_layers) for i in range(num_layers)]
    content_loss_ratios = [0 for i in range(num_layers)]
    content_loss_ratios[num_layers - 1] = 1.0
    print "In run optimization function"
    result = None # result of optimization
    with tf.Graph().as_default():
        # build graph variable input
        initial_value = np.random.randn(1, 1, NUM_SAMPLES, NUM_CHANNELS).astype(np.float32)*1e-3
        if starting == CONTENT:
            print "Initializing with content - only paying attention to content loss"
            initial_value = content_input_tf
            content_loss_ratios = [0 for i in range(num_layers)]
        x = tf.Variable(initial_value, name="x")
        input_to_next = x
        x_content_output = []
	for i in range(num_layers):
            print "Evaluating layer {} of CNN".format(i)
	    kernel_name = "kernel" + str(i+1)
	    layer_name = "conv" + str(i+1)
	    kernel_tf = tf.constant(random_kernels[i], name=kernel_name, dtype='float32')
	    conv = tf.nn.conv2d(
		    input_to_next,
		    kernel_tf,
		    strides = STRIDES,
		    padding = PADDING,
		    name = layer_name)
            print conv.get_shape(), ": conv shape"
            pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[1,1], strides=2)
	    net = tf.nn.relu(pool)
            print net.get_shape(), ": net shape"
	    x_content_output.append(net)
	    input_to_next = net
            
        x_style_output = []
        for i in range(num_layers):
            x_style_output.append(get_style_features(x_content_output[i], num_filters, True)) # True - this is a tensor

        # get the style and content loss
        content_loss_per_layer = []
        style_loss_per_layer = []
        for i in range(num_layers):
            content_loss_per_layer.append( get_content_loss(x_content_output[i], content_refs[i]) )
            style_loss_per_layer.append( get_style_loss( x_style_output[i], style_refs[i]) )

        total_style_loss = sum([style_loss_per_layer[i] * style_loss_ratios[i] for i in range(num_layers)])
        total_content_loss = sum([content_loss_per_layer[i] * content_loss_ratios[i] for i in range(num_layers)])

        loss = total_style_loss + total_content_loss
	opt = tf.contrib.opt.ScipyOptimizerInterface( loss, method='L-BFGS-B', options = {'maxiter': ITERATIONS})
	print "Started opt"

	with tf.Session() as sess:
			
 		sess.run(tf.initialize_all_variables())
        	print('Started optimization.')
		opt.minimize(sess)

        	print 'Final loss:', loss.eval()
        	result = x.eval()
		return result
        
    
def main(CONTENT_FILENAME, STYLE_FILENAME, OUTPUT_FILENAME, num_filters, num_layers, filter_width, starting):
#Runs the optimization given the content and style references, and output file to write to.
#Configurable: num_filters, num_layers, filter width size (filter height is 1)
    # 1: load the spectograms - phase is important for recovery later
    content_spectogram, phase = read_audio_spectrum(CONTENT_FILENAME)
    style_spectogram, phase = read_audio_spectrum(STYLE_FILENAME)
    assert(np.shape(style_spectogram) == np.shape(content_spectogram)) # otherwise need to truncate one
    print "GOT SPECTOGRAM REPRESENTATIONS"

    # 2: turn spectogram into input we can use for the neural net model
    content_input_tf = np.ascontiguousarray(content_spectogram.T[None,None,:,:])
    style_input_tf = np.ascontiguousarray(style_spectogram.T[None,None,:,:])

    # 3: get random kernels for random weight initialization of the layers
    random_kernels = initialize_kernels(num_layers, num_filters, filter_width)
    print "Initialized random kernels"

    # 4: go through the model for the content and style
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        x = tf.placeholder('float32', [1,1,NUM_SAMPLES,NUM_CHANNELS], name="x")
	tf_layers = [x]
        print "STARTING RUNNING GRAPH FOR REF"
	for i in range(num_layers):
            kernel_name = "kernel" + str(i+1)
            layer_name = "conv" + str(i+1)
            print "Optimizing for layer {}".format(i)
	    kernel_tf = tf.constant(random_kernels[i], name=kernel_name, dtype='float32')
            print "shape of x: {}".format(x.get_shape())
            print "shape of kernel: {}".format(np.shape(random_kernels[i]))
	    conv = tf.nn.conv2d(
		    tf_layers[len(tf_layers)-1],
		    kernel_tf,
		    strides = STRIDES,
		    padding = PADDING,
		    name = layer_name)
            print conv.get_shape(), ": shaape of output conv layer from {}".format(i)
            pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[1,2], strides=[2,2])
	    net = tf.nn.relu(pool)
            print net.get_shape(), ": shaape of output pool and relu layer from {}".format(i)
	    tf_layers.append(net)
        tf_layers = tf_layers[1:]
	# 5: run the model for the content reference and style references
	content_output = run_model(x, tf_layers, content_input_tf, sess)
        print "Got output for content reference"
        style_output = run_model(x, tf_layers, style_input_tf, sess)
        style_grams = []
        for style_vec in style_output:
            print "I: {}, shape of the conv layer: {}".format(i, np.shape(content_output[i]))
            style_grams.append(get_style_features(style_vec, num_filters, False)) # is not a tensor - is a constant
	print "Got style reference output"
	# 6: code in optimization code
	"""resulting_audio = run_optimization(random_kernels, content_output, style_grams, num_filters, num_layers, filter_width, starting, content_input_tf)

	# 7: reconstruct the final audio
	a = np.zeros_like(content_spectogram)
	a[:NUM_CHANNELS,:] = np.exp(resulting_audio[0,0].T) - 1
	# This code is supposed to do phase reconstruction
	p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
	for i in range(500):
    		S = a * np.exp(1j*p)
    		x = librosa.istft(S)
    		p = np.angle(librosa.stft(x, N_FFT))

	librosa.output.write_wav(OUTPUT_FILENAME, x, phase)"""

        
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-c','--content', help='Path to content filename', required=True)
	parser.add_argument('-s','--style', help='Path to style filename', required=True)
	parser.add_argument('-o', '--output', help='Path to output wave file to write to', required = True)
	parser.add_argument('-f', '--filters', help="Number filters", required=True, type=int)
	parser.add_argument('-l', '--layers', help="Number layers", required=True, type=int)
	parser.add_argument('-fs', '--filtersize', help="Size of filters", required=True, type=int)
	parser.add_argument('-st', '--starting', help='Starting variables', choices=["random", "content"], required=True)
	args = vars(parser.parse_args())

	main(args['content'], args['style'], args['output'], args['filters'], args['layers'], args['filtersize'], args['starting'])


