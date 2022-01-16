"""
Project: Pyramidal Edge-Maps and Attention Based Guided Thermal Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import tensorflow as tf
from tensorflow.python.ops import array_ops

from custom_vgg16 import *


def rec_loss(output, gt):
	return tf.losses.absolute_difference(gt, output)


def image_gradients(image):
	if image.get_shape().ndims != 4:
		raise ValueError('image_gradients expects a 4D tensor '
		                 '[batch_size, h, w, d], not %s.', image.get_shape())

	image_shape = array_ops.shape(image)
	batch_size, height, width, depth = array_ops.unstack(image_shape)
	dy = image[:, 1:, :, :] - image[:, :-1, :, :]
	dx = image[:, :, 1:, :] - image[:, :, :-1, :]
	shape = array_ops.stack([batch_size, 1, width, depth])
	dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
	dy = array_ops.reshape(dy, image_shape)
	shape = array_ops.stack([batch_size, height, 1, depth])
	dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
	dx = array_ops.reshape(dx, image_shape)
	return dy, dx


def gradient_loss(output, gt):
	g1 = image_gradients(output)
	g2 = image_gradients(gt)
	return tf.losses.absolute_difference(g1, g2)


def convert_to_vgg(output, gt):
	data_dict = loadWeightsData('../vgg16.npy')
	output = tf.concat([output, output, output], axis=3)
	gt = tf.concat([gt, gt, gt], axis=3)

	vgg_c = custom_Vgg16(output, data_dict=data_dict)

	T_features1 = vgg_c.conv3_2
	vgg_c = custom_Vgg16(gt, data_dict=data_dict)
	I_features1 = vgg_c.conv3_2

	return T_features1, I_features1
