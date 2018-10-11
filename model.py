import tensorflow as tf
import get_data as gd
import numpy as np
import os
from urllib import urlretrieve
import matplotlib.pyplot as plt
import tarfile

# lambda for convolution function
conv = lambda inp, filt, str, pad, id : tf.nn.conv2d(input   = inp,
													 filter  = filt,
													 strides = [1, str, str, 1],
													 padding = pad,
													 name    = id)
class cifar:
	def __init__(self, sess, f, c_layers, fc_u, fc_layers, s, mp, mp_s, c,
		length, tr_sample, te_sample, va_sample, batch_size, epochs,
		lr, dropout, batch_norm, continue_training, reg_factor,
		init_mean, init_stddev, seed, p_size):
		
		"""
		sess : TF session in use
		f : list of filter sizes
		c_layers : number of convolutional layers
		fc_u : number of units in fully connected layers
		fc_layers : number of fully connected layers
		s : list of filter strides in convolutional layers
		mp : list of max pool filter sides
		mp_s : list of max pool filter strides
		c : list of channels in each convolutional layers

		length : side length of each image
		tr_sample : number of samples in training set
		te_sample : number of samples in testing set
		va_sample : number of samples in validation set (made separately)
		batch_size : batch size to be used
		epochs : number of epochs to be trained for
		lr : learning rate for the optimizer
		dropout : whether to use dropout or not
		batch_norm : whether to use batch norm or not
		continue_training : whether to continue training from saved model
		add_disturbance : whether to add disturbance or not
		"""

		self.sess = sess
		self.LENGTH = length
		self.tr_sample = tr_sample
		self.te_sample = te_sample
		self.va_sample = va_sample
		self.batch_size = batch_size
		self.epochs = epochs
		self.lr = lr
		self.dropout = dropout
		self.batch_norm = batch_norm
		self.continue_training = continue_training
		self.checkpoint_dir = './checkpoint'
		self.class_names = ['aeroplane',
							'automobile',
							'bird',
							'cat',
							'deer',
							'dog',
							'frog',
							'horse',
							'ship',
							'truck']

		assert (len(f) == c_layers), "Number of filter sides and number of layers do not match"
		assert (len(c) == c_layers), "Number of channels and number of layers do not match"
		assert (len(s) == c_layers), "Number of filter strides and number of layers do not match"
		assert (len(mp_s) == c_layers), "Number of max pool strides and number of layers do not match"
		assert (len(mp) == c_layers), "Number of max pool filter side and number of layers do not match"
		assert (len(fc_u) == fc_layers), "Number of FC layers and number of FC layer units do not match"

		self.c_layers = c_layers
		self.fc_layers = fc_layers
		self.f = f
		self.c = c
		self.fc_u = fc_u
		self.s = s
		self.mp_s = mp_s
		self.mp = mp
		self.mean = init_mean
		self.stddev = init_stddev
		self.seed = seed
		self.p_size = p_size

		self.model_name = "cifar"
		self.model_dir = self.checkpoint_dir+'/'+str(self.batch_size)+'_'+str(self.c_layers)+'_'+str(self.fc_layers)

		self.train_data = np.zeros([5, 10000, 32, 32, 3])
		self.train_label = np.zeros([5, 10000])
		self.occlude = False
		self.reg_factor = reg_factor
		self.create_model()
		self.get_training_data()

	def get_patch(self):
		a1 = np.random.randint(0, 27)
		a2 = np.random.randint(0, 27)
		self.patch = np.ones([32, 32])
		if self.occlude:
			self.patch[a1:a1+3, a2:a2+3] = 0
		return self.patch.reshape([1, 32, 32, 1])

	def mask(self, n1, n2, n3, n4, f):
		mask = np.ones([n1, n2, n3, n4])
		mask[:, :, :, f] = 0
		return tf.convert_to_tensor(mask, dtype=tf.float32)

	def get_training_data(self):
		for i in range(5):
			if not os.path.isfile('./cifar-10-python.tar.gz'):
				print "[!] File not found"
				print "[*] Downloading"
				fname, headers = urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', './cifar-10-python.tar.gz')
				if (fname.endswith('tar.gz')):
					tar = tarfile.open(fname, 'tar.gz')
					tar.extractall()
					tar.close()
					print "[*] Extracted"
					self.file_loc = "./cifar-10-batches-py/"

			data_dict = gd.unpickle(self.file_loc+"data_batch_"+str(i+1))
			input_data = data_dict[b'data']
			input_label = data_dict[b'labels']
			self.train_label[i] = np.asarray(input_label)
			input_data = (input_data.astype(np.float32)/255.0 - 0.5)/0.5
			self.patch = self.get_patch()
			img_R = input_data[:,0:1024].reshape((-1,32, 32,1))*self.patch#+np.random.normal(scale=0.01, size=[32, 32, 1])
			img_G = input_data[:,1024:2048].reshape((-1,32, 32,1))*self.patch#+np.random.normal(scale=0.01, size=[32, 32, 1])
			img_B = input_data[:,2048:3072].reshape((-1,32, 32,1))*self.patch#+np.random.normal(scale=0.01, size=[32, 32, 1])
			self.train_data[i] = np.concatenate((img_R,img_G,img_B),3)

	def get_rotated_data(self):
		for i in range(5):
			data_dict = gd.unpickle(self.file_loc+"data_batch_"+str(i+1))
			input_data = data_dict[b'data']
			input_label = data_dict[b'labels']
			self.train_label[i] = np.asarray(input_label)
			input_data = (input_data.astype(np.float32)/255.0 - 0.5)/0.5
			if np.random.randint(2):
				img_R = np.transpose(input_data[:,0:1024].reshape((-1,32, 32,1)), [0, 2, 1, 3])
				img_G = np.transpose(input_data[:,1024:2048].reshape((-1,32, 32,1)), [0, 2, 1, 3])
				img_B = np.transpose(input_data[:,2048:3072].reshape((-1,32, 32,1)), [0, 2, 1, 3])
			else:
				img_R = input_data[:,0:1024].reshape((-1,32, 32,1))
				img_G = input_data[:,1024:2048].reshape((-1,32, 32,1))
				img_B = input_data[:,2048:3072].reshape((-1,32, 32,1))
			self.train_data[i] = np.concatenate((img_R,img_G,img_B),3)

	def get_noisy_data(self):
		for i in range(5):
			data_dict = gd.unpickle(self.file_loc+"data_batch_"+str(i+1))
			input_data = data_dict[b'data']
			input_label = data_dict[b'labels']
			self.train_label[i] = np.asarray(input_label)
			input_data = (input_data.astype(np.float32)/255.0 - 0.5)/0.5
			img_R = input_data[:,0:1024].reshape((-1,32, 32,1))+np.random.normal(scale=0.01, size=[32, 32, 1])
			img_G = input_data[:,1024:2048].reshape((-1,32, 32,1))+np.random.normal(scale=0.01, size=[32, 32, 1])
			img_B = input_data[:,2048:3072].reshape((-1,32, 32,1))+np.random.normal(scale=0.01, size=[32, 32, 1])
			self.train_data[i] = np.concatenate((img_R,img_G,img_B),3)

	def get_occluded_data(self, p):
		data_dict = gd.unpickle(self.file_loc+"test_batch")
		input_data = data_dict[b'data']
		input_label = data_dict[b'labels']
		self.test_label = np.asarray(input_label)
		input_data = (input_data.astype(np.float32)/255.0 - 0.5)/0.5
		img_R = input_data[:,0:1024].reshape((-1,32, 32,1))
		img_R[:, (p/(33-self.p_size)):(p/(33-self.p_size))+self.p_size, (p%(33-self.p_size)):(p%(33-self.p_size))+self.p_size, 0] = 0.5
		img_G = input_data[:,1024:2048].reshape((-1,32, 32,1))
		img_G[:, (p/(33-self.p_size)):(p/(33-self.p_size))+self.p_size, (p%(33-self.p_size)):(p%(33-self.p_size))+self.p_size, 0] = 0.5
		img_B = input_data[:,2048:3072].reshape((-1,32, 32,1))
		img_B[:, (p/(33-self.p_size)):(p/(33-self.p_size))+self.p_size, (p%(33-self.p_size)):(p%(33-self.p_size))+self.p_size, 0] = 0.5
		self.test_data = np.concatenate((img_R,img_G,img_B),3)

	def get_testing_data(self):
		data_dict = gd.unpickle(self.file_loc+"test_batch")
		input_data = data_dict[b'data']
		input_label = data_dict[b'labels']
		self.test_label = np.asarray(input_label)
		input_data = (input_data.astype(np.float32)/255.0 - 0.5)/0.5
		img_R = input_data[:,0:1024].reshape((-1,32, 32,1))
		img_G = input_data[:,1024:2048].reshape((-1,32, 32,1))
		img_B = input_data[:,2048:3072].reshape((-1,32, 32,1))
		self.test_data = np.concatenate((img_R,img_G,img_B),3)

	def get_data_prepared(self, index, data_type):
		if data_type == "train":
			b = int(index*self.batch_size/10000)
			if ((index+1)*self.batch_size)%10000 != 0:
				images = self.train_data[b, (index*self.batch_size)%10000:((index+1)*self.batch_size)%10000]
				labels = self.train_label[b, (index*self.batch_size)%10000:((index+1)*self.batch_size)%10000]
			else:
				images = self.train_data[b, (index*self.batch_size)%10000:]
				labels = self.train_label[b, (index*self.batch_size)%10000:]
		elif data_type == "validate":
			images = self.validate_data[index*self.batch_size:(index+1)*self.batch_size]
			labels = self.validate_label[index*self.batch_size:(index+1)*self.batch_size]
		elif data_type == "test":
			images = self.test_data[index*self.batch_size:(index+1)*self.batch_size]
			labels = self.test_label[index*self.batch_size:(index+1)*self.batch_size]

		return images, labels

	def create_model(self):
		tf.set_random_seed(0)

		self.img_train = tf.placeholder(tf.float32, shape=[self.batch_size, self.LENGTH, self.LENGTH, 3],
			name='img_input')
		self.img_label = tf.placeholder(tf.int64, shape=[self.batch_size], name="img_label")

		self.conv = []
		self.conv_ = []

		self.regul = tf.contrib.layers.l2_regularizer(scale=self.reg_factor)

		for i in range(self.c_layers):
			with tf.variable_scope("w"+str(i)):
				if i == 0:
					temp_conv = tf.layers.conv2d(self.img_train,
											filters=self.c[i],
											kernel_size=self.f[i],
											strides=(self.s[i], self.s[i]),
											padding="VALID",
											activation=tf.nn.relu,
											kernel_initializer=tf.truncated_normal_initializer(mean=self.mean,
																								stddev=self.stddev/np.sqrt(self.c[i]*self.f[i]*self.f[i]*3),
																								seed=self.seed),
											kernel_regularizer=self.regul,
											name='conv'+str(i))
				else:
					temp_conv = tf.layers.conv2d(self.conv[i-1],
											filters=self.c[i],
											kernel_size=self.f[i],
											strides=(self.s[i], self.s[i]),
											padding="VALID",
											activation=tf.nn.relu,
											kernel_initializer=tf.truncated_normal_initializer(mean=self.mean,
																								stddev=self.stddev/np.sqrt(self.c[i]*self.f[i]*self.f[i]*self.c[i-1]),
																								seed=self.seed),
											kernel_regularizer=self.regul,
											name='conv'+str(i))
			self.conv_.append(temp_conv)

			'''
			# label: filter_mod

			# provide values in places with <> (remove <> also)

			if i == <layer_num>:
				temp_conv = tf.multiply(temp_conv, self.mask(self.batch_size, <side length of conv>, <side length of conv>, <number of channels in conv>, <filter_index>))
			
			'''
			temp_conv = tf.nn.max_pool(temp_conv, [1, self.mp[i], self.mp[i], 1],
										[1, self.mp_s[i], self.mp_s[i], 1], 'VALID')

			if self.batch_norm:
				temp_conv = tf.contrib.layers.batch_norm(temp_conv)
			
			if not self.dropout:
				self.conv.append(temp_conv)
			else:
				self.conv.append(tf.layers.dropout(temp_conv))

		temp_fc = tf.reshape(self.conv[self.c_layers-1], [self.batch_size, -1])
		self.fc = []
		self.fc.append(temp_fc)

		for i in range(self.fc_layers):
			with tf.variable_scope("fc"+str(i)):
				if i != self.fc_layers-1:
					self.fc.append(tf.layers.dense(self.fc[i],
													self.fc_u[i],
													kernel_initializer=tf.truncated_normal_initializer(mean=self.mean,
																										stddev=self.stddev,
																										seed=self.seed),
													activation=tf.nn.relu,
													kernel_regularizer=self.regul))
				else:
					self.fc.append(tf.layers.dense(self.fc[i],
													10,
													kernel_initializer=tf.truncated_normal_initializer(mean=self.mean,
																										stddev=self.stddev,
																										seed=self.seed),
													activation=tf.nn.softmax,
													kernel_regularizer=self.regul))

		self.one_hot = tf.one_hot(self.img_label, 10)
		self.loss       = tf.losses.softmax_cross_entropy(onehot_labels=self.one_hot,logits=self.fc[self.fc_layers])+tf.reduce_mean(self.reg_loss)
		self.loss_graph = tf.summary.scalar("Softmax error", self.loss)
		self.optim      = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		self.acc, self.acc_op = tf.metrics.accuracy(labels=tf.argmax(self.one_hot, 1),
			predictions=tf.argmax(self.fc[self.fc_layers],1))

		self.saver = tf.train.Saver(max_to_keep=5)

	def train_model(self):
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())
		self.graph_data = tf.summary.merge_all() # for TensorBoard
		self.writer     = tf.summary.FileWriter('./logs', self.sess.graph)

		if self.continue_training:
			if self.load():
				print "[*] Model loaded successfully"
			else:
				print "[!] Model could not be loaded"
		
		print("Training on images")

		load_noisy = True
		load_rotated = True
		for i in range(self.epochs):
			avg_error = 0.0
			for j in range(self.tr_sample/self.batch_size):
				batch_x, batch_y = self.get_data_prepared(j, "train")
				_, l, l_graph = self.sess.run([self.optim, self.loss, self.loss_graph],
					feed_dict= {self.img_train: batch_x, self.img_label: batch_y})
				avg_error += l
				# self.writer.add_summary(l_graph, i*(self.tr_sample/self.batch_size)+j)
			avg_error /= self.tr_sample/self.batch_size
			print "Epoch {}/{} Iteration {} Loss {}".format(i, self.epochs, j, avg_error)

			if (i+1) % 10 == 0:
				self.save(i+1) # saving new model

			if i > 30:
				self.occlude = True
				self.get_training_data()

			if i > 100 and load_noisy:
				self.get_noisy_data()
				load_noisy = False

			if i > 400 and load_rotated:
				self.get_rotated_data()
				load_rotated = False

	def save(self, counter):
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

		self.saver.save(self.sess, self.model_dir+'/'+self.model_name, global_step=counter)

	def load(self):
		checkpoint = tf.train.get_checkpoint_state(self.model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			checkpt_name = os.path.basename(checkpoint.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(self.model_dir, checkpt_name))
			return True
		else:
			return False

	'''
	# not required if you don't have a validation set

	def validate_model(self):
		j = np.random.randint(self.va_sample/self.batch_size)
		batch_x, batch_y = self.get_data_prepared(j, "validate")
		output, _, l = self.sess.run([self.acc, self.acc_op, self.loss],
			feed_dict= {self.img_train: batch_x, self.img_label: batch_y})
		output = self.sess.run(self.acc,
			feed_dict= {self.img_train: batch_x, self.img_label: batch_y})

		print "Validation loss : {} Validation accuracy : {}".format(l, output)
	'''

	def test_model(self):
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		if self.load():
			print "[*] Model loaded successfully"
		else:
			print "[!] Model could not be loaded"

		net_accuracy = 0
		count = 0

		self.get_testing_data()

		for j in range(self.te_sample/self.batch_size):
			batch_x, batch_y = self.get_data_prepared(j, "test")
			output, _ = self.sess.run([self.acc, self.acc_op],
				feed_dict= {self.img_train: batch_x, self.img_label: batch_y})
			output = self.sess.run(self.acc,
				feed_dict= {self.img_train: batch_x, self.img_label: batch_y})
			net_accuracy += output
			count += 1

		net_accuracy = net_accuracy/count
		print "Net accuracy", net_accuracy

	def show_misclassified(self):
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		if self.load():
			print "[*] Model loaded successfully"
		else:
			print "[!] Model could not be loaded"

		if not os.path.exists('./misclassified'):
			os.makedirs('./misclassified')

		self.get_testing_data()

		batch_x, batch_y = self.get_data_prepared(0, "test") # for one batch
		output = self.sess.run(self.fc[self.fc_layers],
			feed_dict= {self.img_train: batch_x, self.img_label: batch_y})
		
		for i in range(self.batch_size):
			if np.argmax(output[i]) != batch_y[i]:
				np.save('./misclassified/img_'+self.class_names[np.argmax(output[i])]+'_'+self.class_names[batch_y[i]], batch_x[i])
				print "misclassified. image number", i, "true", self.class_names[batch_y[i]], "predicted", self.class_names[np.argmax(output[i])]
			else:
				print "correctly classified. image number", i

	def occlusion_sensitivity(self):
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		correct_output = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 14, 19, 20, 23]

		prob_variation = np.zeros([15, (33-self.p_size), (33-self.p_size)])

		if self.load():
			print "[*] Model loaded successfully"
		else:
			print "[!] Model could not be loaded"

		if not os.path.exists('./prob_variation'):
			os.makedirs('./prob_variation')

		for p in range((33-self.p_size)*(33-self.p_size)):
			self.get_occluded_data(p)
			batch_x, batch_y =  self.get_data_prepared(0, "test") # for one batch
			output = self.sess.run(self.fc[self.fc_layers],
				feed_dict= {self.img_train: batch_x, self.img_label: batch_y})
			
			for i in range(len(correct_output)):
				prob_variation[i, p/(33-self.p_size), p%(33-self.p_size)] = output[correct_output[i], batch_y[0]]
				print "True label for", correct_output[i], self.class_names[batch_y[correct_output[i]]]
			print "{}/{}".format(p, (33-self.p_size)*(33-self.p_size))

		for i in range(len(correct_output)):
			plt.imshow(prob_variation[i], cmap='hot')
			plt.axis('off')
			plt.savefig('./prob_variation/'+str(self.p_size)+'_'+str(correct_output[i])+'.png', bbox_inches='tight')
			plt.clf()
			plt.imshow(batch_x[correct_output[i]])
			plt.axis('off')
			plt.savefig('./prob_variation/'+str(self.p_size)+'_'+str(correct_output[i])+'_true.png', bbox_inches='tight')
			plt.clf()

	def filter_analysis(self):
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		if self.load():
			print "[*] Model loaded successfully"
		else:
			print "[!] Model could not be loaded"

		if not os.path.exists('./max_response'):
			os.makedirs('./max_response')
		if not os.path.exists('./max_respo_patch'):
			os.makedirs('./max_respo_patch')

		self.get_testing_data()

		# choosing filters to analyse
		# 2 filters from each layer
		filters = np.array([[32, 10],
							[1, 34],
							[14, 43],
							[12, 100],
							[32, 102]])

		image_index = np.zeros_like(filters)
		max_value = np.zeros_like(filters)
		filter_pos = np.zeros([5, 2, 2], dtype=np.int32)

		for j in range(self.te_sample/self.batch_size):
			batch_x, batch_y = self.get_data_prepared(j, "test")
			f1, f2, f3, f4, f5 = self.sess.run([self.conv_[0], self.conv_[1], self.conv_[2], self.conv_[3], self.conv_[4]],
				feed_dict= {self.img_train: batch_x, self.img_label: batch_y})

			# the receptive field should be calculated manually.
			# these values are for the architecture mentioned in the README file
			for im in range(self.batch_size):
				if np.amax(f1[im, :, :, filters[0, 0]]) > max_value[0, 0]:
					image_index[0, 0] = im
					max_value[0, 0] = np.amax(f1[im, :, :, filters[0, 0]])
					filter_pos[0, 0] = np.array([np.argmax(f1[im, :, :, filters[0, 0]])/28, np.argmax(f1[im, :, :, filters[0, 0]])%28])
				if np.amax(f1[im, :, :, filters[0, 1]]) > max_value[0, 1]:
					image_index[0, 1] = im
					max_value[0, 1] = np.amax(f1[im, :, :, filters[0, 1]])
					filter_pos[0, 1] = np.array([np.argmax(f1[im, :, :, filters[0, 1]])/28, np.argmax(f1[im, :, :, filters[0, 1]])%28])

				if np.amax(f2[im, :, :, filters[1, 0]]) > max_value[1, 0]:
					image_index[1, 0] = im
					max_value[1, 0] = np.amax(f2[im, :, :, filters[1, 0]])
					filter_pos[1, 0] = np.array([np.argmax(f2[im, :, :, filters[1, 0]])/23, np.argmax(f2[im, :, :, filters[1, 0]])%23])
				if np.amax(f2[im, :, :, filters[1, 1]]) > max_value[1, 1]:
					image_index[1, 1] = im
					max_value[1, 1] = np.amax(f2[im, :, :, filters[1, 1]])
					filter_pos[1, 1] = np.array([np.argmax(f2[im, :, :, filters[1, 1]])/23, np.argmax(f2[im, :, :, filters[1, 1]])%23])
				
				if np.amax(f3[im, :, :, filters[2, 0]]) > max_value[2, 0]:
					image_index[2, 0] = im
					max_value[2, 0] = np.amax(f3[im, :, :, filters[2, 0]])
					filter_pos[2, 0] = np.array([np.argmax(f3[im, :, :, filters[2, 0]])/7, np.argmax(f3[im, :, :, filters[2, 0]])%7])
				if np.amax(f3[im, :, :, filters[2, 1]]) > max_value[2, 1]:
					image_index[2, 1] = im
					max_value[2, 1] = np.amax(f3[im, :, :, filters[2, 1]])
					filter_pos[2, 1] = np.array([np.argmax(f3[im, :, :, filters[2, 1]])/7, np.argmax(f3[im, :, :, filters[2, 1]])%7])
				
				if np.amax(f4[im, :, :, filters[3, 0]]) > max_value[3, 0]:
					image_index[3, 0] = im
					max_value[3, 0] = np.amax(f4[im, :, :, filters[3, 0]])
					filter_pos[3, 0] = np.array([np.argmax(f4[im, :, :, filters[3, 0]])/5, np.argmax(f4[im, :, :, filters[3, 0]])%5])
				if np.amax(f4[im, :, :, filters[3, 1]]) > max_value[3, 1]:
					image_index[3, 1] = im
					max_value[3, 1] = np.amax(f4[im, :, :, filters[3, 1]])
					filter_pos[3, 1] = np.array([np.argmax(f4[im, :, :, filters[3, 1]])/5, np.argmax(f4[im, :, :, filters[3, 1]])%5])
				
				if np.amax(f5[im, :, :, filters[4, 0]]) > max_value[4, 0]:
					image_index[4, 0] = im
					max_value[4, 0] = np.amax(f5[im, :, :, filters[4, 0]])
					filter_pos[4, 0] = np.array([np.argmax(f5[im, :, :, filters[4, 0]])/3, np.argmax(f5[im, :, :, filters[4, 0]])%3])
				if np.amax(f5[im, :, :, filters[4, 1]]) > max_value[4, 1]:
					image_index[4, 1] = im
					max_value[4, 1] = np.amax(f5[im, :, :, filters[4, 1]])
					filter_pos[4, 1] = np.array([np.argmax(f5[im, :, :, filters[4, 1]])/3, np.argmax(f5[im, :, :, filters[4, 1]])%3])

				print "{}/{} {}/{}".format(j, self.te_sample/self.batch_size, im, self.batch_size)

			for r in range(5):
				for c in range(2):
					img = batch_x[image_index[r, c]]
					plt.imshow(img)
					plt.axis('off')
					plt.savefig('./max_response/'+str(r)+'_'+str(c)+'_'+str(j)+'_true.png')
					plt.clf()

			for f in range(2):
				# filter maps in layer 5
				img = batch_x[image_index[4, f]]
				img_patch = img[filter_pos[4, f, 0]*2:filter_pos[4, f, 0]*2+27, filter_pos[4, f, 1]*2:filter_pos[4, f, 1]*2+27]
				plt.imshow(img_patch)
				plt.axis('off')
				plt.savefig('./max_respo_patch/4_'+str(f)+'_'+str(j)+'.png')

				# filter maps in layer 4
				img = batch_x[image_index[3, f]]
				img_patch = img[filter_pos[3, f, 0]*2:filter_pos[3, f, 0]*2+24, filter_pos[3, f, 1]*2:filter_pos[3, f, 1]*2+24]
				plt.imshow(img_patch)
				plt.axis('off')
				plt.savefig('./max_respo_patch/3_'+str(f)+'_'+str(j)+'.png')

				# filter maps in layer 3
				img = batch_x[image_index[2, f]]
				img_patch = img[filter_pos[2, f, 0]*2:filter_pos[2, f, 0]*2+19, filter_pos[2, f, 1]*2:filter_pos[2, f, 1]*2+19]
				plt.imshow(img_patch)
				plt.axis('off')
				plt.savefig('./max_respo_patch/2_'+str(f)+'_'+str(j)+'.png')

				# filter maps in layer 2
				img = batch_x[image_index[1, f]]
				img_patch = img[filter_pos[1, f, 0]:filter_pos[1, f, 0]+10, filter_pos[1, f, 1]:filter_pos[1, f, 1]+10]
				plt.imshow(img_patch)
				plt.axis('off')
				plt.savefig('./max_respo_patch/1_'+str(f)+'_'+str(j)+'.png')

				# filter maps in layer 1
				img = batch_x[image_index[0, f]]
				img_patch = img[filter_pos[0, f, 0]:filter_pos[0, f, 0]+5, filter_pos[0, f, 1]:filter_pos[0, f, 1]+5]
				plt.imshow(img_patch)
				plt.axis('off')
				plt.savefig('./max_respo_patch/0_'+str(f)+'_'+str(j)+'.png')

	def filter_modification(self):
		# this is specifically for the suggested architecture
		# also, uncomment label:filter_mod in self.create_model()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		if self.load():
			print "[*] Model loaded successfully"
		else:
			print "[!] Model could not be loaded"

		# choosing filters to analyse
		# 2 filters from each layer
		filters = np.array([[32, 10],
							[1, 34],
							[14, 43],
							[12, 100],
							[32, 102]])

		image_index = np.zeros_like(filters)
		max_value = np.zeros_like(filters)

		self.get_testing_data()

		if not os.path.exists('./new_misclassified'):
			os.makedirs('./new_misclassified')

		batch_x, batch_y = self.get_data_prepared(0, "test")
		output = self.sess.run(self.fc[self.fc_layers],
			feed_dict= {self.img_train: batch_x, self.img_label: batch_y})
		
		incorrectly = [2,3,24,25,32,35,36,40,52,57,58,59,61,62,63,65,68,70,71,76,78,84,85,86,87,91,99]
		for i in range(self.batch_size):
			if np.argmax(output[i]) == batch_y[i]:
				print "correctly classified 0. image number", i
			else:
				print "misclassified 0. image number", i
				if i not in incorrectly:
					img = batch_x[i]
					plt.imshow(img)
					plt.axis('off')
					plt.savefig('./new_misclassified/0_0_0_'+str(i)+'.png')
					print i, output[i, batch_y[i]]

		batch_x, batch_y = self.get_data_prepared(1, "test")
		output = self.sess.run(self.fc[self.fc_layers],
			feed_dict= {self.img_train: batch_x, self.img_label: batch_y})

		incorrectly = [0,12,16,17,18,19,20,25,28,35,38,39,47,49,56,58,62,68,72,78,80,84,88,89,90,95,98]
		for i in range(self.batch_size):
			if np.argmax(output[i]) == batch_y[i]:
				print "correctly classified 1. image number", i
			else:
				print "misclassified 1. image number", i
				if i not in incorrectly:
					img = batch_x[i]
					plt.imshow(img)
					plt.axis('off')
					plt.savefig('./new_misclassified/0_0_1_'+str(i)+'.png')
					print i, output[i, batch_y[i]]