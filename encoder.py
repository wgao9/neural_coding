import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import commpy.channelcoding.convcode as cc
import sys
#from commpy.utilities import *

k = int(sys.argv[1])
n = 2*k+4
training_size = 10000
testing_size = 1000
learning_rate = 5
steps = 10000
SNR = float(sys.argv[2])
sigma = 10**(-0.05*SNR)

def generate_data(m,knn):
	if knn <= 0:
		data_1 = np.random.randint(2,size=(m,k))
		data_2 = np.random.randint(2,size=(m,k))
		return data_1, data_2

	data_1 = np.random.randint(2,size=(m,k))
	data_2 = copy.deepcopy(data_1)
	flip_indices = np.random.randint(k,size=(m,knn))
	for i in range(m):
		for j in range(knn):
			data_2[i,flip_indices[i,j]] = 1 - data_2[i,flip_indices[i,j]]
	return data_1, data_2

def conv_encoder(message_bits):
	generator_matrix = np.array([[05, 07]])
	M = np.array([2])
	trellis = cc.Trellis(M, generator_matrix)
	return 2*cc.conv_encode(np.asarray(message_bits), trellis)-1

#Model Parameters
m = np.linspace(k,n,num=2).astype(int)

#Input
x0_1 = tf.placeholder(tf.float32,[None, k])
x0_2 = tf.placeholder(tf.float32, [None, k])
truey_1 = tf.placeholder(tf.float32, [None, n])
truey_2 = tf.placeholder(tf.float32, [None, n])
#Layer 1
We_1 = tf.Variable(tf.random_normal([m[0],m[1]]))
be_1 = tf.Variable(tf.random_normal([m[1]]))
'''
#Layer 2
We_2 = tf.Variable(tf.random_normal([m[1],m[2]]))
be_2 = tf.Variable(tf.random_normal([m[2]]))
#Layer 3
We_3 = tf.Variable(tf.random_normal([m[2],m[3]]))
be_3 = tf.Variable(tf.random_normal([m[3]]))
#Layer 4
We_4 = tf.Variable(tf.random_normal([m[3],m[4]]))
be_4 = tf.Variable(tf.random_normal([m[4]]))
'''

#Model
#Layer 1
y1_1 = 2*tf.nn.sigmoid(tf.matmul(x0_1, We_1) + be_1)-1
y1_2 = 2*tf.nn.sigmoid(tf.matmul(x0_2, We_1) + be_1)-1
x1_1 = np.sqrt(m[1])*tf.nn.l2_normalize(y1_1, dim=1)
x1_2 = np.sqrt(m[1])*tf.nn.l2_normalize(y1_2, dim=1)
'''
#Layer 2
y2_1 = 2*tf.nn.sigmoid(tf.matmul(x1_1, We_2) + be_2)-1
y2_2 = 2*tf.nn.sigmoid(tf.matmul(x1_2, We_2) + be_2)-1
x2_1 = np.sqrt(m[2])*tf.nn.l2_normalize(y2_1, dim=1)
x2_2 = np.sqrt(m[2])*tf.nn.l2_normalize(y2_2, dim=1)
#Layer 3
y3_1 = 2*tf.nn.sigmoid(tf.matmul(x2_1, We_3) + be_3)-1
y3_2 = 2*tf.nn.sigmoid(tf.matmul(x2_2, We_3) + be_3)-1
x3_1 = np.sqrt(m[3])*tf.nn.l2_normalize(y3_1, dim=1)
x3_2 = np.sqrt(m[3])*tf.nn.l2_normalize(y3_2, dim=1)
#Layer 4
y4_1 = 2*tf.nn.sigmoid(tf.matmul(x3_1, We_4) + be_4)-1
y4_2 = 2*tf.nn.sigmoid(tf.matmul(x3_2, We_4) + be_4)-1
x4_1 = np.sqrt(m[4])*tf.nn.l2_normalize(y4_1, dim=1)
x4_2 = np.sqrt(m[4])*tf.nn.l2_normalize(y4_2, dim=1)
'''

y_1 = y1_1
y_2 = y1_2

#Customized Loss Function	
loss_qfunc = tf.log(tf.reduce_mean(2**(k-1)*tf.erfc(tf.norm(y_1-y_2, axis=1)/(2*np.sqrt(2)*sigma))))
loss_conv = 0.125*tf.reduce_mean(tf.square(y_1 - truey_1)) + 0.125*tf.reduce_mean(tf.square(y_2 - truey_2))
optimizer = tf.train.AdadeltaOptimizer(learning_rate)
train = optimizer.minimize(loss_conv)

#Training and Testing Data
train_1, train_2 = generate_data(training_size,1)
test_1, test_2 = generate_data(testing_size,1)
train_y_1, train_y_2 = np.zeros((training_size,n)), np.zeros((training_size,n))
test_y_1, test_y_2 = np.zeros((testing_size,n)), np.zeros((testing_size,n))
for i in range(training_size):
	train_y_1[i] = conv_encoder(train_1[i])
	train_y_2[i] = conv_encoder(train_2[i])
for i in range(testing_size):
	test_y_1[i] = conv_encoder(test_1[i])
	test_y_2[i] = conv_encoder(test_2[i])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

conv_distance = [np.linalg.norm(conv_encoder(test_1[i]) - conv_encoder(test_2[i])) for i in range(len(test_1))]
conv_loss = np.log(np.mean(2**(k-1)*sp.special.erfc(conv_distance/(2*np.sqrt(2)*sigma))))

#Training
for i in range(steps):
	if i%100 == 0:
		train_loss = sess.run(loss_conv, {x0_1: train_1, x0_2: train_2, truey_1: train_y_1, truey_2: train_y_2})
		test_loss = sess.run(loss_conv, {x0_1: test_1, x0_2: test_2, truey_1: test_y_1, truey_2: test_y_2})
		average_distance = sess.run(tf.reduce_mean(tf.norm(y_1-y_2, axis=1)), {x0_1: test_1, x0_2: test_2})
		print("step: %s, training loss: %s, testing loss: %s, goal loss: %s, av_dist: %s, goal_dist: %s"%(i, train_loss, test_loss, conv_loss, average_distance, np.mean(conv_distance)))
		print(sess.run(y_1, {x0_1: train_1, x0_2: train_2, truey_1: train_y_1, truey_2: train_y_2}))
		print(test_y_1)
	sess.run(train,{x0_1: train_1, x0_2: train_2, truey_1: train_y_1, truey_2: train_y_2})
'''
#Save Weights
saver = tf.train.Saver({'We_1': We_1, 'be_1': be_1, 'We_2': We_2, 'be_2': be_2, 'We_3': We_3, 'be_3': be_3, 'We_4': We_4, 'be_4': be_4})
saver_path = saver.save(sess, "/home/wgao9/encoder/tmp/model_k=%s_n=%s_4layer.ckpt" %(k,n))
print("Model saved in file: %s" % saver_path)
'''
