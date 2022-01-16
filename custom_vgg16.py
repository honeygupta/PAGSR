import inspect
import os

import numpy as np
import tensorflow as tf
import vgg16

VGG_MEAN = [103.939, 116.779, 123.68]


def loadWeightsData(vgg16_npy_path='../densenet/'):
	if vgg16_npy_path is None:
		path = inspect.getfile(Vgg16)
		path = os.path.abspath(os.path.join(path, os.pardir))
		path = os.path.join(path, "vgg16.npy")
		vgg16_npy_path = path
		print(vgg16_npy_path)
	return np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()


class custom_Vgg16(vgg16.Vgg16):
	# Input should be an rgb image [batch, height, width, 3]
	# values scaled [0, 1]

	def __init__(self, rgb, data_dict, train=False):
		# It's a shared weights data and used in various
		# member functions.

		self.data_dict = data_dict

		rgb_scaled = (rgb + 1) * 127.5
		# Convert RGB to BGR
		print("RGB_SCALED_SHAPE = " + str(tf.shape(rgb_scaled)))
		red, green, blue = tf.split(rgb_scaled, 3, 3)

		bgr = tf.concat([blue - VGG_MEAN[0],
		                 green - VGG_MEAN[1],
		                 red - VGG_MEAN[2]],
		                3)

		self.conv1_1 = self.conv_layer(bgr, "conv1_1")
		self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
		self.pool1 = self.max_pool(self.conv1_2, 'pool1')

		self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
		self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
		self.pool2 = self.max_pool(self.conv2_2, 'pool2')

		self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
		self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
		self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
		self.pool3 = self.max_pool(self.conv3_3, 'pool3')

		self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
		self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
		self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
		self.pool4 = self.max_pool(self.conv4_3, 'pool4')

		self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
		self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
		self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
		self.pool5 = self.max_pool(self.conv5_3, 'pool5')

	def debug(self):
		pass
