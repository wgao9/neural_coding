import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import commpy.channelcoding.convcode as cc
import sys
#from commpy.utilities import *

k = 10
n = 2*k+4
learning_rate_conv = 1
learning_rate_qfunc = 0.01
momentum = 0.75
steps = 5000
SNR = 0.0
sigma = 10**(-0.05*SNR)
m = 20
var = 0.1
batch_size = 64
training_batches = 10000
testing_batches = 100
beta = 0
rnn_type = sys.argv[1]
training_type = sys.argv[2]
num_of_layers = int(sys.argv[3])

def generate_data(batch_size, fb):
	x = np.random.randint(2,size=(batch_size,k,1))
	y = np.zeros((batch_size,k,2))
	for i in range(batch_size):
		y[i] = conv_encoder(x[i].reshape(k), fb)[:2*k].reshape((k,2))
	return x, y

def generate_data_knn(batch_size, knn, fb, flip=-1):
	y_1 = np.zeros((batch_size,k,2))
	y_2 = np.zeros((batch_size,k,2))
	if knn <= 0:
		x_1 = np.random.randint(2,size=(batch_size,k,1))
		x_2 = np.random.randint(2,size=(batch_size,k,1))
		for i in range(batch_size):
			y_1[i] = conv_encoder(x_1[i].reshape(k), fb)[:2*k].reshape((k,2))
			y_2[i] = conv_encoder(x_2[i].reshape(k), fb)[:2*k].reshape((k,2))
		return x_1, x_2, y_1, y_2
	
	x_1 = np.random.randint(2,size=(batch_size,k,1))
	x_2 = copy.deepcopy(x_1)
	flip_indices = np.random.randint(k,size=(batch_size,knn))
	if flip >= 0:
		flip_indices = flip+np.zeros((batch_size,knn))
	for i in range(batch_size):
		for j in range(knn):
			x_2[i,flip_indices[i,j],0] = 1 - x_2[i,flip_indices[i,j],0]
		y_1[i] = conv_encoder(x_1[i].reshape(k), fb)[:2*k].reshape((k,2))
		y_2[i] = conv_encoder(x_2[i].reshape(k), fb)[:2*k].reshape((k,2))
	return x_1, x_2, y_1, y_2
	
def conv_encoder(message_bits, fb):
	generator_matrix = np.array([[07, 05]])
	M = np.array([2])
	if fb == True:
		trellis = cc.Trellis(M, generator_matrix, feedback=7)
	else:
		trellis = cc.Trellis(M, generator_matrix)
	return 2*cc.conv_encode(np.asarray(message_bits), trellis)-1

def lstm_cell(m):
	return tf.nn.rnn_cell.LSTMCell(m)

def gru_cell(m):
	return tf.nn.rnn_cell.GRUCell(m)

def rnn_cell(m):
	return tf.nn.rnn_cell.BasicRNNCell(m)

#Model Inputs
x_1 = tf.placeholder(tf.float32, [batch_size, k, 1])
x_2 = tf.placeholder(tf.float32, [batch_size, k, 1])
x = tf.placeholder(tf.float32, [batch_size, k, 1])
true_y = tf.placeholder(tf.float32, [batch_size, k, 2])

#Model
if rnn_type == "LSTM":
	cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(m) for _ in range(num_of_layers)])
elif rnn_type == "GRU":
	cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell(m) for _ in range(num_of_layers)])
else:
	cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(m) for _ in range(num_of_layers)])

softmax_W = tf.Variable(tf.random_normal([m,2], 0.0, var))
softmax_b = tf.Variable(tf.random_normal([2], 0.0, var))

#Model
state = cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)
state_1 = cell.zero_state(batch_size, dtype=tf.float32)
outputs_1, state_1 = tf.nn.dynamic_rnn(cell, x_1, initial_state=state_1, dtype=tf.float32)
state_2 = cell.zero_state(batch_size, dtype=tf.float32)
outputs_2, state_2 = tf.nn.dynamic_rnn(cell, x_2, initial_state=state_2, dtype=tf.float32)

hat_z = 2*tf.nn.sigmoid(tf.tensordot(outputs, softmax_W, axes=[[2],[0]]) + softmax_b)-1
#hat_mean, hat_var = tf.nn.moments(hat_z, axes=[0], keep_dims=True)
#hat_y = np.sqrt(2*k)*tf.div(tf.subtract(hat_z, hat_mean), tf.sqrt(hat_var))
hat_y = hat_z
z_1 = 2*tf.nn.sigmoid(tf.tensordot(outputs_1, softmax_W, axes=[[2],[0]]) + softmax_b)-1
z_2 = 2*tf.nn.sigmoid(tf.tensordot(outputs_2, softmax_W, axes=[[2],[0]]) + softmax_b)-1
mean_1, var_1 = tf.nn.moments(z_1, axes=[1,2], keep_dims=True)
y_1 = tf.div(tf.subtract(z_1, mean_1), tf.sqrt(var_1))
mean_2, var_2 = tf.nn.moments(z_2, axes=[1,2], keep_dims=True)
y_2 = tf.div(tf.subtract(z_2, mean_2), tf.sqrt(var_2))

#Losses
loss_ce = tf.reduce_mean(-tf.log(tf.clip_by_value(tf.abs(0.5*(true_y+hat_y)),1e-15,1.0)))
loss_l2 = tf.reduce_mean(tf.square(true_y - hat_y))
loss_01 = tf.reduce_mean(tf.round(tf.abs(true_y - hat_y)))
loss_qfunc = tf.log(tf.reduce_mean(2**(k-1)*tf.erfc(tf.norm(y_1-y_2, axis=(1,2))/(2*np.sqrt(2)*sigma))))
avg_dist = tf.reduce_mean(tf.norm(y_1 - y_2, axis=(1,2)))
avg_dists = [tf.reduce_mean(tf.norm(y_1[:,i]-y_2[:,i], axis=(1))) for i in range(k)]

#Regularizer
reg = sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
reg_l1 = tf.reduce_mean(tf.norm(y_1-y_2, ord=1, axis=(1,2)))

#Customized Loss Function	
optimizer_conv = tf.train.MomentumOptimizer(learning_rate_conv, momentum)
train_ce = optimizer_conv.minimize(loss_ce + beta*reg_l1)
optimizer_qfunc = tf.train.MomentumOptimizer(learning_rate_qfunc, momentum)
train_qfunc = optimizer_qfunc.minimize(loss_qfunc + beta*reg_l1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#Training
if training_type == "conv":
	train_loss_l2, train_loss_ce, train_loss_01 = [], [], []
	for i in range(steps):
		train_x, train_y = generate_data(batch_size, fb=True)
		sess.run(train_ce,{x: train_x, true_y: train_y})
		train_loss_l2.append(sess.run(loss_l2, {x: train_x, true_y: train_y}))
		train_loss_ce.append(sess.run(loss_ce, {x: train_x, true_y: train_y}))
		train_loss_01.append(sess.run(loss_01, {x: train_x, true_y: train_y}))
		if i%25 == 0:
			print("step: %s, l2 loss: %.6f, ce loss: %.6f, 01 loss: %.6f"%(i, np.mean(train_loss_l2), np.mean(train_loss_ce), np.mean(train_loss_01)))
			if np.mean(train_loss_ce) < 1e-3:
				break
			train_loss_l2, train_loss_ce, train_loss_01 = [], [], []
elif training_type == "qfunc":
	train_loss, train_avg_dist, goal_dists, goal_loss, goal_avg_dist = [], [], [], [], []
	thr = [0.0, 1.5, 0.5, 1.5]
	for i in range(steps):
		train_x_1, train_x_2, train_y_1, train_y_2 = generate_data_knn(batch_size, knn=1, fb=False)
		sess.run(train_qfunc, {x_1: train_x_1, x_2: train_x_2})
		train_loss.append(sess.run(loss_qfunc, {x_1: train_x_1, x_2: train_x_2}))
		train_avg_dist.append(sess.run(avg_dist, {x_1: train_x_1, x_2: train_x_2}))
		goal_dists.append(np.linalg.norm(train_y_1 - train_y_2, axis=(1,2)))
		goal_loss.append(np.log(np.mean(2**(k-1)*sp.special.erfc(goal_dists/(2*np.sqrt(2)*sigma)))))
		goal_avg_dist.append(np.mean(goal_dists))
		if i%25 == 0:
			print("step: %s, train loss: %.6f. train dist: %.6f, goal loss: %.6f, goal dist: %.6f" % (i, np.mean(train_loss), np.mean(train_avg_dist), np.mean(goal_loss), np.mean(goal_avg_dist)))
			train_x_1_f, train_x_2_f, _, _ = generate_data_knn(batch_size, knn=1, fb=False, flip=0)
			print(sess.run(avg_dists, {x_1: train_x_1_f, x_2: train_x_2_f}))
			if np.mean(train_loss) < np.mean(goal_loss)-thr[num_of_layers]:
				break
			train_loss, train_avg_dist, goal_dists, goal_loss, goal_avg_dist = [], [], [], [], []
'''
#For k = 10 only
if k <= 10:
	x_all = []
	y_all = []
	for mes in range(2**k):
		message = [int(x) for x in bin(mes)[2:].zfill(k)]
		x_all = np.concatenate((x, message), axis=0)
		if (mes+1)%batch_size == 0:
			y_mes = sess.run(y_1, {x_1: x_all})
			y_all = np.concatenate((y_all, y_mes), axis=0)
			x_all = []
	distance = [100]
	for mes in range(1,2**k):
		distance[mes] = np.linalg.norm(y_all[0]-y_all[mes], axis=(1,2))
	print distance, np.min(distance), np.argmin(distance)
'''
'''
#Save Weights
saver = tf.train.Saver()
saver_path = saver.save(sess, "/home/wgao9/encoder/tmp/"+rnn_type+"_k=%s_n=%s_"%(k,n)+training_type+"_%slayers_nn.ckpt"%num_of_layers)
print("Model saved in file: %s" % saver_path)
'''
#Save data
train_x, _ = generate_data(batch_size, fb=True)
test_x, _ = generate_data(batch_size, fb=True)
train_y = sess.run(y_1, {x_1: train_x})
test_y = sess.run(y_2, {x_2: train_x})
for i in range(training_batches-1):
	train_x_i, _ = generate_data(batch_size, fb=True)
	train_y_i = sess.run(y_1, {x_1: train_x_i})
	train_x = np.concatenate((train_x, train_x_i), axis=0)
	train_y = np.concatenate((train_y, train_y_i), axis=0)
	if i%25==0:
		print("Storing training data number %s"%i)

for i in range(testing_batches-1):
	test_x_i, _ = generate_data(batch_size, fb=True)
	test_y_i = sess.run(y_1, {x_1: test_x_i})
	test_x = np.concatenate((test_x, test_x_i), axis=0)
	test_y = np.concatenate((test_y, test_y_i), axis=0)
	if i%25==0:
		print("Storing testing data number %s"%i)

outfile = "data/"+rnn_type+"_k=%s_m=%s"%(k,m)+"_%slayers_"%num_of_layers+training_type+"_l1reg.npz"
np.savez(outfile, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
print("Data saved in file: %s" % outfile)

