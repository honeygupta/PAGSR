"""
Project: Pyramidal Edge-Maps and Attention Based Guided Thermal Super-Resolution
Author: Honey Gupta (hn.gpt1@gmail.com)

"""
import json
import os
from datetime import datetime

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import click
import tensorflow as tf

from skimage.io import imsave
import datasets
import data_loader
import model
import losses
import tensorflow.contrib.slim as slim

slim = tf.contrib.slim


class Network:

	def __init__(self, output_root_dir, to_restore,
	             base_lr, max_step, dataset_name, checkpoint_dir, do_flipping):
		current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

		self._output_dir = os.path.join(output_root_dir, current_time)
		self._images_dir = os.path.join(self._output_dir, 'imgs')
		self._to_restore = to_restore
		self._base_lr = base_lr
		self._max_step = max_step
		self._dataset_name = dataset_name
		self._checkpoint_dir = checkpoint_dir
		self._do_flipping = do_flipping
		self._num_imgs_to_save = 1

	def model_setup(self, mode):

		if mode == 'train':
			factor = 1
		if mode == 'test':
			factor = 1
		self.input = tf.placeholder(
			tf.float32, [
				model.BATCH_SIZE,
				model.L_IMG_HEIGHT // factor,
				model.L_IMG_WIDTH // factor,
				model.IMG_CHANNELS
			], name="input")
		self.gt = tf.placeholder(
			tf.float32, [
				model.BATCH_SIZE,
				model.H_IMG_HEIGHT // factor,
				model.H_IMG_WIDTH // factor,
				model.IMG_CHANNELS
			], name="gt")
		self.edges = tf.placeholder(
			tf.float32, [
				model.BATCH_SIZE,
				model.G_IMG_HEIGHT // factor,
				model.G_IMG_WIDTH // factor,
				6
			], name="edges")

		self.global_step = slim.get_or_create_global_step()
		self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

		inputs = {

			'image_in': self.input,
			'image_gt': self.gt,
			'image_edges': self.edges
			# 'image_re': self.resized

		}
		outputs = model.get_outputs(inputs)

		self.output = outputs['output']

	def compute_losses(self):

		rec_loss = 10 * losses.rec_loss(self.output, self.gt)
		T_features1, I_features1 = losses.convert_to_vgg(self.output, self.gt)

		feature_loss = 0.0001 * losses.rec_loss(T_features1,
		                                        I_features1)

		grad_loss = losses.gradient_loss(self.output, self.gt)
		g_loss = rec_loss + feature_loss + grad_loss

		optimizer = tf.train.AdamOptimizer(self.learning_rate)

		self.model_vars = tf.trainable_variables()

		g_vars = [var for var in self.model_vars if 'g_' in var.name]

		self.g_trainer = optimizer.minimize(g_loss, var_list=g_vars)

		# Summary variables for tensorboard
		self.g_loss_summ = tf.summary.scalar("g_loss", g_loss)

	def save_images(self, sess, epoch):

		if not os.path.exists(self._images_dir):
			os.makedirs(self._images_dir)

		names = ['input_', 'gt_', 'edges0_', 'edges1_', 'edges2_', 'edges3_', 'edges4_', 'edges5_', 'output_']

		with open(os.path.join(
				self._output_dir, 'epoch_' + str(epoch) + '.html'
		), 'w') as v_html:
			for i in range(0, self._num_imgs_to_save):
				print("Saving image {}/{}".format(i, self._num_imgs_to_save))

				inputs = sess.run(self.inputs)

				output = sess.run(self.output, feed_dict={
					self.input: inputs['image_in'],
					self.gt: inputs['image_gt'],
					self.edges: inputs['image_edges']
				})
				tensors = [inputs['image_in'], inputs['image_gt'], inputs['image_edges'][:, :, :, 0],
				           inputs['image_edges'][
				           :, :, :, 1],
				           inputs['image_edges'][:, :, :, 2], inputs['image_edges'][:, :, :, 3],
				           inputs['image_edges'][:, :, :, 4],
				           inputs['image_edges'][:, :, :, 5], output]

				for batch in range(model.BATCH_SIZE):
					for name, tensor in zip(names, tensors):
						image_name = os.path.split(inputs['filename'][batch].decode())[-1] + '_' + name + str(
							i) + 'batchid_' + str(batch) + ".png"
						imsave(os.path.join(self._images_dir, image_name),
						       ((np.squeeze(tensor[batch]) + 1) * 127.5).astype(np.uint8)
						       )
						v_html.write(
							"<img src=\"" +
							os.path.join('imgs', image_name) + "\">"
						)
					v_html.write("<br>")

	def train(self):
		"""Training Function."""

		self.inputs = data_loader.load_data(self._dataset_name, mode='train')
		# Build the network
		self.model_setup(mode='train')

		# Loss function calculations
		self.compute_losses()

		# Initializing the global variables
		init = (tf.global_variables_initializer(), tf.local_variables_initializer())
		saver = tf.train.Saver(max_to_keep=10)

		max_images = datasets.DATASET_TO_SIZES[self._dataset_name] // model.BATCH_SIZE

		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
			sess.run(init)

			# Restore the model to run the model from last checkpoint
			if self._to_restore:
				chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
				print('restoring saved checkpoint : ' + str(chkpt_fname))
				saver.restore(sess, chkpt_fname)

			writer = tf.summary.FileWriter(self._output_dir)
			if not os.path.exists(self._output_dir):
				os.makedirs(self._output_dir)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			# Training Loop
			for epoch in range(sess.run(self.global_step), self._max_step):
				print("In the epoch ", epoch)
				if epoch % 1 == 0:
					saver.save(sess, os.path.join(
						self._output_dir, "pegsr"), global_step=epoch)

				# Dealing with the learning rate as per the epoch number
				if epoch < 100:
					curr_lr = self._base_lr
				elif epoch >= 100 and epoch < 200:
					curr_lr = self._base_lr - \
					          self._base_lr * (epoch - 100) / 100
				else:
					curr_lr = self._base_lr * 0.01

				if epoch % 1 == 0:
					self.save_images(sess, epoch)

				for i in range(0, max_images):
					print("Processing batch {}/{}".format(i, max_images))

					inputs = sess.run(self.inputs)

					# Optimizing the G_A network
					_, summary_str = sess.run(
						[self.g_trainer,
						 self.g_loss_summ],
						feed_dict={
							self.input:
								inputs['image_in'],
							self.gt:
								inputs['image_gt'],
							self.edges:
								inputs['image_edges'],
							self.learning_rate: curr_lr
						}
					)

					writer.add_summary(summary_str, epoch * max_images + i)

					writer.flush()

				sess.run(tf.assign(self.global_step, epoch + 1))

			coord.request_stop()
			coord.join(threads)
			writer.add_graph(sess.graph)

	def test(self):
		"""Test Function."""
		print("Testing the results")

		self.inputs = data_loader.load_data(self._dataset_name, mode='test')
		print(self.inputs['image_edges'])

		self.model_setup(mode='test')
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		# with tf.device('/device:GPU:0'):
		with tf.Session(config=config) as sess:
			sess.run(init)

			chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
			print(chkpt_fname)
			saver.restore(sess, chkpt_fname)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			self._num_imgs_to_save = datasets.DATASET_TO_SIZES[
				self._dataset_name]
			self.save_images(sess, 0)

			coord.request_stop()
			coord.join(threads)


@click.command()
@click.option('--to_train',
              type=click.INT,
              default=0,
              help='Whether it is train or false.')
@click.option('--log_dir',
              type=click.STRING,
              default='output/',
              help='Where the data is logged to.')
@click.option('--config_filename',
              type=click.STRING,
              default='configs/test.json',
              help='The name of the configuration file.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='checkpoint/',
              help='The name of the train/test split.')
def main(to_train, log_dir, config_filename, checkpoint_dir):
	if not os.path.isdir(log_dir):
		os.makedirs(log_dir)

	with open(config_filename) as config_file:
		config = json.load(config_file)

	to_restore = (to_train == 2)
	base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
	max_step = int(config['max_step']) if 'max_step' in config else 200
	dataset_name = str(config['dataset_name'])
	do_flipping = bool(config['do_flipping'])

	newmodel = Network(log_dir,
	                   to_restore, base_lr, max_step,
	                   dataset_name, checkpoint_dir, do_flipping)

	if to_train > 0:
		newmodel.train()
	else:
		newmodel.test()


if __name__ == '__main__':
	main()
