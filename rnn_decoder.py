import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import commpy.channelcoding.convcode as cc
import sys
#from commpy.utilities import *
import itertools

k = 100
n = 2*k+4
learning_rate_l2 = 1
learning_rate_ce = 2
momentum = 0.75
steps = 2000
m = 200
var = 0.1
batch_size = 64
beta = 2e-10
filename = sys.argv[1]
rnn_type = sys.argv[2]
train_SNR = 0.0
train_sigma = 10**(-0.05*train_SNR)
test_SNR = float(sys.argv[3])
test_sigma = 10**(-0.05*test_SNR)

def BER(data, decoded_data):
	wrong, total = 0.0, len(data)*len(data[0])*len(data[0][0])
	for row, col, ind in itertools.product(range(len(data)), range(len(data[0])), range(len(data[0][0]))):
		if abs(data[row][col][ind] - decoded_data[row][col][ind]) > 0.5:
			wrong += 1.0
	p = wrong/total
	return p, np.sqrt(p*(1-p)/total)

def BLER(data, decoded_data):
	wrong, total = 0.0, len(data)
	for row in range(len(data)):
		for col, ind in itertools.product(range(len(data[0])), range(len(data[0][0]))):
			if abs(data[row][col][ind] - decoded_data[row][col][ind]) > 0.5:
				wrong += 1.0
				break
	p = wrong/total
	return p, np.sqrt(p*(1-p)/total)

#Model Inputs
x = tf.placeholder(tf.float32, [batch_size, k, 1])
y = tf.placeholder(tf.float32, [batch_size, k, 2])

#Model
if rnn_type == "LSTM":
	cell_fw_0 = tf.nn.rnn_cell.LSTMCell(m)
	cell_bw_0 = tf.nn.rnn_cell.LSTMCell(m)
	cell_fw_1 = tf.nn.rnn_cell.LSTMCell(m)
	cell_bw_1 = tf.nn.rnn_cell.LSTMCell(m)
elif rnn_type == "GRU":
	cell_fw_0 = tf.nn.rnn_cell.GRUCell(m)
	cell_bw_0 = tf.nn.rnn_cell.GRUCell(m)
	cell_fw_1 = tf.nn.rnn_cell.GRUCell(m)
	cell_bw_1 = tf.nn.rnn_cell.GRUCell(m)
else:
	cell_fw_0 = tf.nn.rnn_cell.BasicRNNCell(m)
	cell_bw_0 = tf.nn.rnn_cell.BasicRNNCell(m)
	cell_fw_1 = tf.nn.rnn_cell.BasicRNNCell(m)
	cell_bw_1 = tf.nn.rnn_cell.BasicRNNCell(m)

beta_0 = tf.Variable(tf.zeros([batch_size, k, 2*m]))
gamma_0 = tf.Variable(tf.ones([batch_size, k, 2*m]))
beta_1 = tf.Variable(tf.zeros([batch_size, k, 2*m]))
gamma_1 = tf.Variable(tf.ones([batch_size, k, 2*m]))

softmax_W = tf.Variable(tf.random_normal([4*m,1], 0.0, var))
softmax_b = tf.Variable(tf.random_normal([1], 0.0, var))
#
sess=tf.Session()
#Model
state_fw_0 = cell_fw_0.zero_state(batch_size, dtype=tf.float32)
state_bw_0 = cell_bw_0.zero_state(batch_size, dtype=tf.float32)
outputs_0, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_0, cell_bw_0, y, initial_state_fw=state_fw_0, initial_state_bw=state_bw_0, scope = 'layer_0', dtype=tf.float32)
output_0 = tf.concat(outputs_0,2)
mean_0, var_0 = tf.nn.moments(output_0, axes=[0], keep_dims=True)
output_0 = tf.nn.batch_normalization(output_0, mean_0, var_0, beta_0, gamma_0, 1e-3)
 
state_fw_1 = cell_fw_1.zero_state(batch_size, dtype=tf.float32)
state_bw_1 = cell_bw_1.zero_state(batch_size, dtype=tf.float32)
outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_1, cell_bw_1, output_0, initial_state_fw=state_fw_1, initial_state_bw=state_bw_1, scope = 'layer_1', dtype=tf.float32)
output_1 = tf.concat(outputs_1,2)
mean_1, var_1 = tf.nn.moments(output_1, axes=[0], keep_dims=True)
output_1 = tf.nn.batch_normalization(output_1, mean_1, var_1, beta_1, gamma_1, 1e-3)

hat_x = tf.nn.sigmoid(tf.tensordot(tf.concat([output_0, output_1],2), softmax_W, axes=[[2],[0]]) + softmax_b)

#Losses
loss_ce = tf.reduce_mean(-tf.log(tf.clip_by_value(tf.abs(1.0-x-hat_x),1e-15,1.0)))
loss_l2 = tf.reduce_mean(tf.square(x - hat_x))
loss_01 = tf.reduce_mean(tf.round(tf.abs(x - hat_x)))

#Regularizer
reg = sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

#Customized Loss Function	
optimizer_l2 = tf.train.MomentumOptimizer(learning_rate_l2+beta*reg, momentum)
optimizer_ce = tf.train.MomentumOptimizer(learning_rate_ce+beta*reg, momentum)
train_l2 = optimizer_l2.minimize(loss_l2)
train_ce = optimizer_ce.minimize(loss_ce)
#sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#Loading Data
outfile = "data/"+filename+".npz"
npzfile = np.load(outfile)
print("Data loaded from file: %s" % outfile)
train_x = npzfile['train_x'] #64000*100*1
training_batches = len(train_x)/batch_size
train_y = npzfile['train_y'] + np.random.normal(0.0,train_sigma,size=[training_batches*batch_size,k,2]) #64000*100*2
test_x = npzfile['test_x'] #6400*100*1
testing_batches = len(test_x)/batch_size
test_y = npzfile['test_y'] + np.random.normal(0.0,test_sigma,size=[testing_batches*batch_size,k,2]) #6400*100*1


#Training
for i in range(steps):
	train_inds = np.random.randint(training_batches*batch_size, size=(batch_size))
	train_x_i = train_x[train_inds]
	train_y_i = train_y[train_inds]
	sess.run(train_l2, {x: train_x_i, y: train_y_i})

	if i%25 == 0:		
		train_loss_l2 = sess.run(loss_l2, {x: train_x_i, y: train_y_i})
		train_loss_ce = sess.run(loss_ce, {x: train_x_i, y: train_y_i})
		test_y_i = test_y[0:batch_size]
		test_hat_x = sess.run(hat_x, {y: test_y_i})
		for ind in range(1,testing_batches):
			test_y_i = test_y[ind*batch_size:(ind+1)*batch_size]
			test_hat_x = np.concatenate((test_hat_x, sess.run(hat_x, {y: test_y_i})), axis = 0)
		ber, ber_std = BER(test_x, test_hat_x)
		bler, bler_std = BLER(test_x, test_hat_x)
		print("step: %s, l2 loss: %.6f, ce loss: %.6f, BER: %.6f+-%.6f, BLER: %.6f+-.%6f"%(i, train_loss_l2, train_loss_ce, ber, ber_std, bler, bler_std))

