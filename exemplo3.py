import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from gmproc import ClientServer, ClientWorker, ServerWorker
import tensorflow as tf
import time

tf.enable_eager_execution()

class MyModel(keras.Model):
	def __init__(self, state_size, action_size):
		super(MyModel, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.dense1 = layers.Dense(4, activation='sigmoid')
		self.fn_out = layers.Dense(action_size)

	def call(self, inputs):
		# Forward pass
		x = self.dense1(inputs)
		o = self.fn_out(x)
		return o

class MyServer(ServerWorker):
	def __init__(self):
		super().__init__()
		self.name = "sharednetwork"
		self.state_size = 2
		self.action_size = 1
		self.model = MyModel(self.state_size, self.action_size)
		self.model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
		self.opt = tf.train.AdamOptimizer(0.001, use_locking=True)

	def start(self):
		print('starting server %s'%(self.name))

	def process(self, id, msg):
		#print("Msg from client %s"%(id))
		if msg is not None:
			grads = [None]*len(msg)
			for i in range(len(msg)):
				grads[i] = tf.convert_to_tensor(msg[i], dtype=tf.float32)
			self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
		return self.model.get_weights()

	def wait(self):
		time.sleep(0.1)

class Client(ClientWorker):
	def __init__(self, name):
		super().__init__()
		self.name = name
		self.state_size = 2
		self.action_size = 1
		self.model = MyModel(self.state_size, self.action_size)
		self.model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
		self.opt = tf.train.AdamOptimizer(0.1, use_locking=True)
		self.started = False
		self.x = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
		self.y = [ [0.0], [1.0], [1.0], [0.0]]
		self.sess = tf.Session()
		self.counter = 0
		self.ep_loss = 0

	def start(self):
		print("starting client %s"%(self.name))


	def print_loss(self):
		loss = self.ep_loss.numpy()
		avg = np.mean(loss)/self.counter
		print("Loss: ", avg)

	def process(self):
		if self.started:
			with tf.GradientTape() as tape:
				tape.watch(self.model.trainable_weights)
				ye = self.model(tf.convert_to_tensor(np.array(self.x), dtype=tf.float32))
				total_loss = tf.keras.losses.MSE(np.array(self.y), ye)
			self.ep_loss += total_loss
			# Calculate local gradients
			grads = tape.gradient(total_loss, self.model.trainable_weights)
			results = []
			for i in range(len(grads)):
				results.append(grads[i].numpy())
			if self.counter % 10 and self.counter > 10:
				print("%d: "%(self.counter), end='')
				self.print_loss()
			self.counter += 1
			return results
		else:
			return None

	def update(self, neww):
		if neww is not None:
			self.model.set_weights(neww)
			self.started = True

	def finish(self):
		print("Training Finished with final loss: ", end='')
		self.print_loss()
		y = self.model(tf.convert_to_tensor(self.x, dtype=tf.float32)).numpy()
		for i, x in enumerate(self.x):
			print(x, " ===> ", y[i])

	def done(self):
		return self.counter > 1000


if __name__=="__main__":
	cs = ClientServer(MyServer())
	cs.add('xor_agent', lambda:Client("xor_agent"))
	cs.run()
