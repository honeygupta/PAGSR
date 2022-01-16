"""
Project: Pyramidal Edge-Maps and Attention Based Guided Thermal Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import tensorflow as tf

import layers

slim = tf.contrib.slim

BATCH_SIZE = 1

H_IMG_HEIGHT = 480
H_IMG_WIDTH = 640

G_IMG_HEIGHT = H_IMG_HEIGHT
G_IMG_WIDTH = H_IMG_WIDTH

L_IMG_HEIGHT = H_IMG_HEIGHT // 8
L_IMG_WIDTH = H_IMG_WIDTH // 8

IMG_CHANNELS = 1

ngf = 32
ndf = 64


def get_outputs(inputs):
	images_in = inputs['image_in']
	images_edges = inputs['image_edges']

	with tf.variable_scope("Model") as scope:
		current_generator = generator
		output = current_generator(images_in, images_edges, name="g_")

	return {
		'output': output,
	}


def preact_conv(inputs, n_filters, kernel_size=[3, 3], stride=1):
	preact = tf.nn.relu(inputs)
	conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None, padding="SAME",
	                   weights_initializer=tf.truncated_normal_initializer(
		                   stddev=0.02), biases_initializer=tf.constant_initializer(0.0), rate=stride)
	return conv


def DenseBlock(stack, n_layers, growth_rate, scope=None, stride=1):
	with tf.name_scope(scope) as sc:
		new_features = []
		for j in range(n_layers):
			layer = preact_conv(stack, growth_rate, stride=stride)
			new_features.append(layer)
			stack = tf.concat([stack, layer], axis=-1)
		new_features = tf.concat(new_features, axis=-1)
		return stack, new_features


def generator(inputA, edges, name="generator"):
	f = 5
	ks = 3
	# padding = "REFLECT"

	n_pool = 3
	growth_rate = 8
	n_layers_per_block = [2, 2, 2, 2, 2, 2, 2]

	with tf.variable_scope(name):
		n_filters = ngf

		# skip_connection_list = []

		o_c1 = layers.general_conv2d(
			inputA, ngf, f, f, 1, 1, 0.02, name="c1", padding="SAME")
		o_c2 = layers.general_conv2d(
			o_c1, ngf, ks, ks, 1, 1, 0.02, "SAME", "c2")

		o_c3 = layers.general_conv2d(
			o_c2, ngf, ks, ks, 1, 1, 0.02, "SAME", "c3")

		o_c4 = layers.general_deconv2d(
			o_c3, ngf, ks, ks, 2, 2, 0.02, "SAME", "c4")

		o_c4_1 = layers.general_conv2d(
			o_c4, ngf, ks, ks, 1, 1, 0.02, "SAME", "c4_1")

		o_c5 = layers.general_deconv2d(
			o_c4_1, ngf, ks, ks, 2, 2, 0.02, "SAME", "c5")

		o_c5_1 = layers.general_conv2d(
			o_c5, ngf, ks, ks, 1, 1, 0.02, "SAME", "c5_1")

		o_c6 = layers.general_deconv2d(
			o_c5_1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c6")

		stack = slim.conv2d(o_c6, ngf, [3, 3], padding='SAME', activation_fn=tf.nn.leaky_relu,
		                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
		                                                                                       mode='FAN_IN',
		                                                                                       uniform=False))

		stack, _ = DenseBlock(stack, n_layers_per_block[0], growth_rate, scope='denseblock%d' % (1))
		n_filters += growth_rate * n_layers_per_block[0]

		out1 = tf.tile(tf.expand_dims(edges[:, :, :, 1], 3), [1, 1, 1, 48])
		stack = tf.add(stack, out1)

		stack, _ = DenseBlock(stack, n_layers_per_block[1], growth_rate, scope='denseblock%d' % (2))
		n_filters += growth_rate * n_layers_per_block[1]

		out2 = tf.tile(tf.expand_dims(edges[:, :, :, 2], 3), [1, 1, 1, 64])
		stack = tf.add(stack, out2)

		stack, _ = DenseBlock(stack, n_layers_per_block[2], growth_rate, scope='denseblock%d' % (3))
		n_filters += growth_rate * n_layers_per_block[1]

		stack, _ = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate,
		                      scope='denseblock%d' % (n_pool + 1))

		out3 = tf.tile(tf.expand_dims(edges[:, :, :, 3], 3), [1, 1, 1, 96])
		stack = tf.add(stack, out3)

		stack, _ = DenseBlock(stack, n_layers_per_block[n_pool + 1], growth_rate,
		                      scope='denseblock%d' % (n_pool + 2))

		out4 = tf.tile(tf.expand_dims(edges[:, :, :, 4], 3), [1, 1, 1, 112])
		stack = tf.add(stack, out4)

		stack, _ = DenseBlock(stack, n_layers_per_block[n_pool + 2], growth_rate,
		                      scope='denseblock%d' % (n_pool + 3))

		out5 = tf.tile(tf.expand_dims(edges[:, :, :, 5], 3), [1, 1, 1, 128])
		stack = tf.add(stack, out5)

		stack, _ = DenseBlock(stack, n_layers_per_block[n_pool + 3], growth_rate,
		                      scope='denseblock%d' % (n_pool + 4))

		net = tf.nn.tanh(
			layers.general_conv2d(stack, 1, 3, 3, 1, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False))

		net = (net + tf.image.resize_images(inputA, tf.shape(net)[1:3], align_corners=True)) / 2.0

		return net
