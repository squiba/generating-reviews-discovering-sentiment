import tensorflow as tf
import numpy as np

import argparse
import configparser
from datetime import datetime
import os

from tqdm import tqdm



class chatbot:
	def __init__(self):
		self.args = None
		self.Model_dir = 'save/model/'
		self.Model_name = 'model'
		self.config_file = 'params.ini'
		self.datapath = 'quora_data/'
		self.tensorboard_dir='save/model/'

	@staticmethod
	def parsearg(args):
		parser = argparse.ArgumentParser()

		training_args = parser.add_argument_group('Trainng options')

		training_args.add_argument('--num_epoch',type=int,default=10,help = 'maximum number of apoch to run')
		training_args.add_argument('--save_every',type=int, default=10,help = 'create a checkpoint after save_every steps')
		training_args.add_argument('--batch_size',type=int,default=10000,help = 'batch size')
		training_args.add_argument('--learning_rate',type = float,default=0.0003,help = 'learning rate')

		return parser.parse_args(args)

	def main(self,args = None):
		self.args = self.parsearg(args)

		x = tf.placeholder(tf.float32,[None,4096*2])
		y = tf.placeholder(tf.int32,[None])

		w = tf.Variable(tf.truncated_normal([4096*2,2], stddev = np.sqrt(0.5)))
		b = tf.Variable(tf.zeros([2]))

		pred = tf.nn.softmax(tf.matmul(x,w) + b)
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits=pred))
		optimizer = tf.train.AdamOptimizer(self.args.learning_rate).minimize(cost)

		#Saving the variables
		save_list = [var for var in tf.global_variables()]
		self.saver = tf.train.Saver(save_list)

		self.sess = tf.Session()
		#Tensorboard
		graphkey_training = tf.GraphKeys()
		with tf.name_scope("Loss"):
			tf.summary.scalar('Training', cost)
		train_sum_op = tf.summary.merge_all()
		run_name = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
		self.summary_writer_op = tf.summary.FileWriter(self.tensorboard_dir+run_name+'/',graph=self.sess.graph)

		self.initializer = tf.initialize_all_variables()

		
		self.sess.run(self.initializer)

		#training
		self.global_step = 0

		for epoch in range(self.args.num_epoch):
			data_gen = self.data_iter()
			for features,labels in tqdm(data_gen,desc='Training'):
				_,c,summary= self.sess.run([optimizer, cost, train_sum_op],feed_dict = {x: features,y: labels})
				self.summary_writer_op.add_summary(summary,self.global_step)

				self.global_step +=1

				if self.global_step % self.args.save_every == 0:
					model = self.Model_dir + self.Model_name + '-' + str(self.global_step) + '.ckpt'
					self.saver.save(self.sess,model)
					tqdm.write("----- Step %d -- Loss %.2f " % (self.global_step, c))



	def data_iter(self):
		for datafile in os.listdir(self.datapath):
			data = np.load(self.datapath+datafile)
			chunks = int(len(data) / self.args.batch_size)
			for i in range(chunks):
				partition = data[self.args.batch_size*i:self.args.batch_size*(i+1)]
				yield partition[:,:-1], partition[:,-1]

if __name__ == "__main__":
	classifier = chatbot()
	classifier.main()