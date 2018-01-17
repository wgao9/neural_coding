import numpy as np
import scipy as sp
import copy
import tensorflow as tf
import commpy.channelcoding.convcode as cc
import sys
from tensorflow.python import pywrap_tensorflow
import itertools

k = 100
n = 204
training_size = 200000
testing_size = 100
SNR = float(sys.argv[4])
sigma = 10**(-0.05*SNR)
learning_rate = 100
steps = 2500
m = 100
batch_size = 64
momentum = 0.7
rnn_type = sys.argv[1]
training_type = sys.argv[2]
num_of_layers = int(sys.argv[3])

def conv_encoder(message_bits):
	generator_matrix = np.array([[05, 07]])
	M = np.array([2])
	trellis = cc.Trellis(M, generator_matrix)
	return 2*cc.conv_encode(np.asarray(message_bits), trellis)-1

def conv_decoder(coded_bits):
	generator_matrix = np.array([[05, 07]])
	M = np.array([2])
	trellis = cc.Trellis(M, generator_matrix)
	tb_depth = 5*(M.sum()+1)
	return cc.viterbi_decode(coded_bits.astype(float), trellis, tb_depth, decoding_type = 'unquantized')

def BER(data, decoded_data):
	wrong = 0.0
	for row, col, ind in itertools.product(range(len(data)), range(len(data[0])), range(len(data[0][0]))):
		if abs(data[row][col][ind] - decoded_data[row][col][ind]) > 0.5:
			wrong += 1.0
	return wrong/(len(data)*len(data[0])*len(data[0][0]))

def BLER(data, decoded_data):
	wrong = 0.0
	for row in range(len(data)):
		for col, ind in itertools.product(range(len(data[0])), range(len(data[0][0]))):
			if abs(data[row][col][ind] - decoded_data[row][col][ind]) > 0.5:
				wrong += 1.0
				break
	return wrong/len(data)

def show_weight(filename):
	reader = pywrap_tensorflow.NewCheckpointReader(filename)
	var_to_shape_map = reader.get_variable_to_shape_map()
	for key in var_to_shape_map:
		print("tensor_name;", key)
		print(reader.get_tensor(key).shape)
		print(reader.get_tensor(key))
	
###################################################################################################
#show_weight("/home/wgao9/encoder/tmp/"+rnn_type+"_k=100_n=204_qfunc_2layers.ckpt")
tf.reset_default_graph()

if num_of_layers == 1:
	filename = "/home/wgao9/encoder/tmp"+rnn_type+"_k=100_n=204_"+training_type+".ckpt"
	if rnn_type == 'GRU':
		candidate_bias = tf.get_variable("rnn/gru_cell/candidate/bias", shape=[m], trainable=False)
		gates_kernel = tf.get_variable("rnn/gru_cell/gates/kernel", shape=[m+1,2*m], trainable=False)
		candidate_kernel = tf.get_variable("rnn/gru_cell/candidate/kernel", shape=[m+1,m], trainable=False)
		gates_bias = tf.get_variable("rnn/gru_cell/gates/bias", shape=[2*m], trainable=False)
		cell = tf.nn.rnn_cell.GRUCell(m, reuse = True)
	elif rnn_type == 'LSTM':
		bias = tf.get_variable("rnn/lstm_cell/bias", shape=[4*m], trainable=False)
		kernel = tf.get_variable("rnn/lstm_cell/kernel", shape=[m+1,4*m], trainable=False)
		cell = tf.nn.rnn_cell.LSTMCell(m, reuse = True)
	else:
		bias = tf.get_variable("rnn/basic_rnn_cell/bias", shape=[m], trainable=False)
		kernel = tf.get_variable("rnn/basic_rnn_cell/kernel", shape=[m+1,m], trainable=False)
		cell = tf.nn.rnn_cell.BasicRNNCell(m, reuse = True)
elif num_of_layers == 2:
	filename="/home/wgao9/encoder/tmp/"+rnn_type+"_k=100_n=204_"+training_type+"_2layers.ckpt"
	if rnn_type == 'GRU':
		candidate_bias_0 = tf.get_variable("rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias", shape=[m], trainable=False)
		gates_kernel_0 = tf.get_variable("rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel", shape=[m+1,2*m], trainable=False)
		candidate_kernel_0 = tf.get_variable("rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel", shape=[m+1,m], trainable=False)
		gates_bias_0 = tf.get_variable("rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias", shape=[2*m], trainable=False)
		cell_0 = tf.nn.rnn_cell.GRUCell(m, reuse = True)
		candidate_bias_1 = tf.get_variable("rnn/multi_rnn_cell/cell_1/gru_cell/candidate/bias", shape=[m], trainable=False)
		gates_kernel_1 = tf.get_variable("rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel", shape=[2*m,2*m], trainable=False)
		candidate_kernel_1 = tf.get_variable("rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel", shape=[2*m,m], trainable=False)
		gates_bias_1 = tf.get_variable("rnn/multi_rnn_cell/cell_1/gru_cell/gates/bias", shape=[2*m], trainable=False)
		cell_1 = tf.nn.rnn_cell.GRUCell(m, reuse = True)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell_0, cell_1])
	elif rnn_type == 'LSTM':
		bias_0 = tf.get_variable("rnn/multi_rnn_cell/cell_0/lstm_cell/bias", shape=[4*m], trainable=False)
		kernel_0 = tf.get_variable("rnn/multi_rnn_cell/cell_0/lstm_cell/kernel", shape=[m+1,4*m], trainable=False)
		cell_0 = tf.nn.rnn_cell.LSTMCell(m, reuse = True)
		bias_1 = tf.get_variable("rnn/multi_rnn_cell/cell_1/lstm_cell/bias", shape=[4*m], trainable=False)
		kernel_1 = tf.get_variable("rnn/multi_rnn_cell/cell_1/lstm_cell/kernel", shape=[2*m,4*m], trainable=False)
		cell_1 = tf.nn.rnn_cell.LSTMCell(m, reuse = True)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell_0, cell_1])
	else:
		bias_0 = tf.get_variable("rnn/multi_rnn_cell/cell_0/basic_rnn_cell/bias", shape=[m], trainable=False)
		kernel_0 = tf.get_variable("rnn/multi_rnn_cell/cell_0/basic_rnn_cell/kernel", shape=[m+1,m], trainable=False)
		cell_0 = tf.nn.rnn_cell.BasicRNNCell(m, reuse = True)
		bias_1 = tf.get_variable("rnn/multi_rnn_cell/cell_1/basic_rnn_cell/bias", shape=[m], trainable=False)
		kernel_1 = tf.get_variable("rnn/multi_rnn_cell/cell_1/basic_rnn_cell/kernel", shape=[2*m,m], trainable=False)
		cell_1 = tf.nn.rnn_cell.BasicRNNCell(m, reuse = True)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell_0, cell_1])
	
	
softmax_W = tf.get_variable("Variable", shape=[m,2], trainable=False)
softmax_b = tf.get_variable("Variable_1", shape=[2], trainable=False)
sess = tf.Session() 
saver = tf.train.Saver()
saver.restore(sess, filename)

#Model Inputs
belief = tf.Variable(tf.random_normal([batch_size, k, 1], 0.0, 0.1))
x = tf.placeholder(tf.float32, [batch_size, k, 1])
true_y = tf.placeholder(tf.float32, [batch_size, k, 2])

#Model
state = cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)
z = 2*tf.nn.sigmoid(tf.tensordot(outputs, softmax_W, axes=[[2],[0]]) + softmax_b)-1
y = np.sqrt(2*k)*tf.nn.l2_normalize(z, dim=(1,2))

hat_x = tf.nn.sigmoid(belief)	
hat_state = cell.zero_state(batch_size, dtype=tf.float32)
hat_outputs, hat_state = tf.nn.dynamic_rnn(cell, hat_x, initial_state=hat_state, dtype=tf.float32)
hat_z = 2*tf.nn.sigmoid(tf.tensordot(hat_outputs, softmax_W, axes=[[2],[0]]) + softmax_b)-1
hat_y = np.sqrt(2*k)*tf.nn.l2_normalize(hat_z, dim=(1,2))

#Loss Function
loss = tf.reduce_mean(tf.square(hat_y-true_y))
optimizer = tf.train.AdagradOptimizer(learning_rate)
train = optimizer.minimize(loss, var_list = [belief])
var_list = [belief]
sess.run(tf.variables_initializer(var_list))
init = tf.variables_initializer([optimizer.get_slot(var, name) for name in optimizer.get_slot_names() for var in var_list])
sess.run(init)

#Generate Data
data = np.random.randint(2, size=(batch_size,k,1))
codeword = sess.run(y, {x: data})
noisy_codeword = codeword + np.random.normal(0.0, sigma, size=(batch_size,k,2))

#Experiment: Init with data
sess.run(belief.assign(4*data-2))
#print data, codeword, noisy_codeword
for i in range(steps):
	if i%50 == 0:
		real_data = sess.run(hat_x, {true_y: noisy_codeword})
		bin_data = 0.5*(np.sign(2*real_data-1)+1)
		real_codeword = sess.run(y, {x: real_data})
		bin_codeword = sess.run(y, {x: bin_data})

		print ("d_a_t_a_ = %s" % data[0].reshape([k])[:6])
		print ("realdata = %s" % real_data[0].reshape([k])[:6])
		print ("bin_data = %s" % bin_data[0].reshape([k])[:6])
		print ("codeword = %s" % noisy_codeword[0].reshape([2*k])[:6])
		print ("realcdwd = %s" % real_codeword[0].reshape([2*k])[:6])
		print ("bin_cdwd = %s" % bin_codeword[0].reshape([2*k])[:6])

		loss_BP_bin = np.mean(np.linalg.norm(noisy_codeword - bin_codeword, axis=1))
		loss_BP_real = np.mean(np.linalg.norm(noisy_codeword - real_codeword, axis=1))
		loss_true = np.mean(np.linalg.norm(noisy_codeword - codeword, axis=1))
		d_bin_real = np.mean(np.linalg.norm(bin_codeword - real_codeword, axis=1))
		d_bin_true = np.mean(np.linalg.norm(bin_codeword - codeword, axis=1))
		d_real_true = np.mean(np.linalg.norm(real_codeword - codeword, axis=1))
		loss_l2 = sess.run(loss, {true_y: noisy_codeword})
		ber = BER(data, bin_data)
		bler = BLER(data, bin_data)
		print("step:%s, loss:%.6f, ber:%.6f, bler:%.6f" % (i, loss_l2, ber, bler))
		print("l_BP_bin:%.4f, l_BP_real:%.4f, l_true:%.4f, d_bin_real:%.4f, d_bin_true:%.4f, d_real_true:%.4f" % (loss_BP_bin, loss_BP_real, loss_true, d_bin_real, d_bin_true, d_real_true))
	
	sess.run(train, {true_y: noisy_codeword})

	#if i%10 == 9:
	#	sess.run(belief.assign(tf.sign(belief)*(tf.abs(belief)+2.0*tf.nn.relu(tf.abs(belief)-1.5))))

