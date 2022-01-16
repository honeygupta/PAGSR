"""
Project: Pyramidal Edge-Maps and Attention Based Guided Thermal Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import tensorflow as tf

import datasets
import model


def random_crop(img1, img2, img3, img4, img5, img6, img7, img8, size, factor):
	img1_temp = tf.cast(tf.image.resize_images(img1, [model.H_IMG_HEIGHT, model.H_IMG_WIDTH], align_corners=True),
	                    tf.uint8)

	combined = tf.concat([img1_temp, img2, img3, img4, img5, img6, img7, img8], axis=2)
	image_shape = tf.shape(img1_temp)
	combined_pad = tf.image.pad_to_bounding_box(
		combined, 0, 0,
		tf.maximum(size[0], image_shape[0]),
		tf.maximum(size[1], image_shape[1]))

	combined_crop = tf.random_crop(
		combined_pad,
		size=tf.concat([size, [1 * 8]],
		               axis=0))

	img1 = combined_crop[:, :, 0]
	img1 = tf.expand_dims(img1, 2)
	return (
		tf.image.resize_images(img1, [model.L_IMG_HEIGHT // factor, model.L_IMG_WIDTH // factor], align_corners=True),
		combined_crop[:, :, 1], combined_crop[:, :, 2], combined_crop[:, :, 3],
		combined_crop[:, :, 4], combined_crop[:, :, 5], combined_crop[:, :, 6], combined_crop[:, :, 7])


def _load_samples(csv_name, image_type):
	filename_queue = tf.train.string_input_producer([csv_name])

	reader = tf.TextLineReader()
	_, csv_filename = reader.read(filename_queue)

	record_defaults = [tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string),
	                   tf.constant([], dtype=tf.string)]

	filename_in, filename_gt, filename_fuse, filename_out1, filename_out2, filename_out3, filename_out4, filename_out5 = \
		tf.decode_csv(csv_filename, record_defaults=record_defaults)

	file_contents_in = tf.read_file(filename_in)
	file_contents_gt = tf.read_file(filename_gt)
	file_contents_fuse = tf.read_file(filename_fuse)
	file_contents_out1 = tf.read_file(filename_out1)
	file_contents_out2 = tf.read_file(filename_out2)
	file_contents_out3 = tf.read_file(filename_out3)
	file_contents_out4 = tf.read_file(filename_out4)
	file_contents_out5 = tf.read_file(filename_out5)

	image_decoded_A = tf.image.decode_png(file_contents_in, channels=model.IMG_CHANNELS)
	image_decoded_B = tf.image.decode_png(file_contents_gt, channels=model.IMG_CHANNELS, dtype=tf.uint8)
	image_decoded_fuse = tf.image.decode_png(file_contents_fuse, channels=1, dtype=tf.uint8)
	image_decoded_out1 = tf.image.decode_png(file_contents_out1, channels=1, dtype=tf.uint8)
	image_decoded_out2 = tf.image.decode_png(file_contents_out2, channels=1, dtype=tf.uint8)
	image_decoded_out3 = tf.image.decode_png(file_contents_out3, channels=1, dtype=tf.uint8)
	image_decoded_out4 = tf.image.decode_png(file_contents_out4, channels=1, dtype=tf.uint8)
	image_decoded_out5 = tf.image.decode_png(file_contents_out5, channels=1, dtype=tf.uint8)

	return image_decoded_A, image_decoded_B, image_decoded_fuse, image_decoded_out1, image_decoded_out2, image_decoded_out3, \
	       image_decoded_out4, image_decoded_out5, filename_gt


def load_data(dataset_name, mode='train'):
	if dataset_name not in datasets.DATASET_TO_SIZES:
		raise ValueError('split name %s was not recognized.'
		                 % dataset_name)

	csv_name = datasets.PATH_TO_CSV[dataset_name]

	image_in, image_gt, image_fuse, image_out1, image_out2, image_out3, image_out4, image_out5, filename = _load_samples(
		csv_name, datasets.DATASET_TO_IMAGETYPE[dataset_name])

	if mode == 'train':
		factor = 0.5
	else:
		factor = 1
	image_in, image_gt, image_fuse, image_out1, image_out2, image_out3, image_out4, image_out5 = random_crop(image_in,
	                                                                                                         image_gt,
	                                                                                                         image_fuse,
	                                                                                                         image_out1,
	                                                                                                         image_out2,
	                                                                                                         image_out3,
	                                                                                                         image_out4,
	                                                                                                         image_out5,
	                                                                                                         [
		                                                                                                         model.H_IMG_HEIGHT // factor,
		                                                                                                         model.H_IMG_WIDTH // factor],
	                                                                                                         factor)

	image_gt = tf.expand_dims(image_gt, axis=2)
	image_fuse = tf.expand_dims(image_fuse, axis=2)
	image_out1 = tf.expand_dims(image_out1, axis=2)
	image_out2 = tf.expand_dims(image_out2, axis=2)
	image_out3 = tf.expand_dims(image_out3, axis=2)
	image_out4 = tf.expand_dims(image_out4, axis=2)
	image_out5 = tf.expand_dims(image_out5, axis=2)

	image_edge = tf.concat([image_fuse, image_out1, image_out2, image_out3, image_out4, image_out5], axis=2)

	image_in = tf.cast(image_in, tf.float32)
	image_gt = tf.cast(image_gt, tf.float32)
	image_edge = tf.cast(image_edge, tf.float32)

	inputs = {
		'image_in': image_in,
		'image_gt': image_gt,
		'image_edges': image_edge,
		'filename': filename}

	inputs['image_in'] = tf.subtract(tf.div(inputs['image_in'], 127.5), 1)
	inputs['image_gt'] = tf.subtract(tf.div(inputs['image_gt'], 127.5), 1)
	inputs['image_edges'] = tf.subtract(tf.div(inputs['image_edges'], 127.5), 1)

	inputs['image_gt'], inputs['image_in'], inputs['image_edges'], inputs['filename'] = tf.train.batch(
		[inputs['image_gt'],
		 inputs['image_in'],
		 inputs['image_edges'], inputs['filename']], model.BATCH_SIZE)

	return inputs
