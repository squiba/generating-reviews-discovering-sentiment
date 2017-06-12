import tensorflow as tf
import numpy as np

import argparse
import configparser
import datetime
import os

from tqdm import tqdm



class chatbot:
	def __init__(self):
		self.args = None
		self.Model_dir = 'save/model'
		self.Model_name = 'model'
		self.config_file = 'params.ini'

	def parsearg(args):
		parser = argparse.ArgumentParser()

		training_args = parser.add_argument_group('Trainng options')

		training_args.add_argument('--num_epoch',type=int,default=10,help='maximum number of apoch to run')
		training_args.add_argument('--save_every',type=int, default=10,'create a checkpoint after save_every steps')
		training_args.add_argument('--batch_size',type=int,default=1000,'batch size')
		training_args.add_argument('--learning_rate',type = float,default=0.003,help='learning rate')

		return parser.parse_args(args)

	def main(self,args = None):
		self.args = self.parsearg(args)

		x = tf.placeholder(tf.float32,[None,4096*2])
		y = tf.placeholder(tf.float32,[None,2])

		w = tf.Variable(tf.truncated_normal([4046*2,2], stddev = np.sqrt(0.5)))
		b = tf.Variable(tf.zeros[2])
		pred = tf.nn.softmax(tf.matmul(x,w) + b)
		cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indicies = 1))
		optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

		#Saving the variables
		save_list = [var for var in tf.global_variables()]
		saver = tf.train.Saver(save_list)

		#Tensorboard
		graphkey_training = tf.GraphKeys()
		with tf.name_scope("Loss"):
			tf.summary.scalar('Training', cost, collection = [graphkey_training])
		train_sum_op = tf.summary.merge_all(key = graphkey_training)
		run_name = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

		summary_writer_op = tf.summary.FileWriter(tensorboard_dir+'/'+run_name+'/',graph=session.graph)
		initializer = tf.initialize_all_variables()

		self.sess = tf.Session()
		sess.run(initializer)



	#training
	global_step = 0

	for epoch in range(training_epochs):
		for batch in nextbatch()
		_,c = sess.run([optimizer, cost],feede_dict = {x: batch[:,0:-1],y: batch[:,-1:]})
		global_step +=1

		if global_step % save_every == 0:
			self.saver.save(sess,"model")
