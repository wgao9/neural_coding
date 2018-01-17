import numpy as np
import scipy as sp
import copy
import tensorflow as tf
import commpy.channelcoding.convcode as cc
import sys

k = 100
n = 204
training_size = 200000
testing_size = 100
SNR = float(sys.argv[1])
sigma = 10**(-0.05*SNR)
learning_rate = 50
steps = 15000


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
	for row in range(len(data)):
		for col in range(len(data[0])):
			if abs(data[row][col] - decoded_data[row][col]) > 0.5:
				wrong += 1.0
	return wrong/(len(data)*len(data[0]))

def BLER(data, decoded_data):
	wrong = 0.0
	for row in range(len(data)):
		for col in range(len(data[0])):
			if abs(data[row][col] - decoded_data[row][col]) > 0.5:
				wrong += 1.0
				break
	return wrong/len(data)

###################################################################################################
#Encoder Model Parameters
m = np.linspace(k,n,num=5).astype(int)
var = 0.1
#Layer 1
We_1 = tf.get_variable("We_1", shape=[m[0],m[1]])
be_1 = tf.get_variable("be_1", shape=[m[1]])
#Layer 2
We_2 = tf.get_variable("We_2", shape=[m[1],m[2]])
be_2 = tf.get_variable("be_2", shape=[m[2]])
#Layer 3
We_3 = tf.get_variable("We_3", shape=[m[2],m[3]])
be_3 = tf.get_variable("be_3", shape=[m[3]])
#Layer 4
We_4 = tf.get_variable("We_4", shape=[m[3],m[4]])
be_4 = tf.get_variable("be_4", shape=[m[4]])

saver = tf.train.Saver()
sess = tf.Session() 
saver.restore(sess, "/home/wgao9/encoder/tmp/model_k=100_n=204_4layer.ckpt")
	
#Input
x0 = tf.placeholder(tf.float32, shape=[testing_size,k])
true_y = tf.placeholder(tf.float32, shape=[testing_size,n])
#Model
belief = tf.Variable(tf.random_normal([testing_size,k],0.0,var))
hat_x0 = tf.nn.sigmoid(belief)
#Layer 1
y1 = 2*tf.nn.sigmoid(tf.matmul(x0, We_1) + be_1)-1
x1 = np.sqrt(m[1])*tf.nn.l2_normalize(y1, dim=1)
hat_y1 = 2*tf.nn.sigmoid(tf.matmul(hat_x0, We_1) + be_1)-1
hat_x1 = np.sqrt(m[1])*tf.nn.l2_normalize(hat_y1, dim=1)
#Layer 2
y2 = 2*tf.nn.sigmoid(tf.matmul(x1, We_2) + be_2)-1
x2 = np.sqrt(m[2])*tf.nn.l2_normalize(y2, dim=1)
hat_y2 = 2*tf.nn.sigmoid(tf.matmul(hat_x1, We_2) + be_2)-1
hat_x2 = np.sqrt(m[2])*tf.nn.l2_normalize(hat_y2, dim=1)
#Layer 1
y3 = 2*tf.nn.sigmoid(tf.matmul(x2, We_3) + be_3)-1
x3 = np.sqrt(m[3])*tf.nn.l2_normalize(y3, dim=1)
hat_y3 = 2*tf.nn.sigmoid(tf.matmul(hat_x2, We_3) + be_3)-1
hat_x3 = np.sqrt(m[3])*tf.nn.l2_normalize(hat_y3, dim=1)
#Layer 2
y4 = 2*tf.nn.sigmoid(tf.matmul(x3, We_4) + be_4)-1
x4 = np.sqrt(m[4])*tf.nn.l2_normalize(y4, dim=1)
hat_y4 = 2*tf.nn.sigmoid(tf.matmul(hat_x3, We_4) + be_4)-1
hat_x4 = np.sqrt(m[4])*tf.nn.l2_normalize(hat_y4, dim=1)

y = x4
hat_y = hat_x4

#Loss Function
loss = tf.reduce_mean(tf.square(hat_y-true_y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss, var_list = [belief])
init = tf.global_variables_initializer()
sess.run(init)

data = np.random.randint(2,size=(testing_size,k))
codeword = sess.run(y, {x0: data})
noisy_codeword = codeword + np.random.normal(0.0, sigma, size=(testing_size,n))

#print data, codeword, noisy_codeword
for i in range(steps):
	if i%100 == 0:
		real_data = sess.run(hat_x0, {true_y: noisy_codeword})
		bin_data = 0.5*(np.sign(2*real_data-1)+1)
		real_codeword = sess.run(y, {x0: real_data})
		bin_codeword = sess.run(y, {x0: bin_data})

		loss_BP_bin = np.mean(np.linalg.norm(noisy_codeword - bin_codeword, axis=1))
		loss_BP_real = np.mean(np.linalg.norm(noisy_codeword - real_codeword, axis=1))
		loss_true = np.mean(np.linalg.norm(noisy_codeword - codeword, axis=1))
		d_bin_real = np.mean(np.linalg.norm(bin_codeword - real_codeword, axis=1))
		d_bin_true = np.mean(np.linalg.norm(bin_codeword - codeword, axis=1))
		d_real_true = np.mean(np.linalg.norm(real_codeword - codeword, axis=1))
		ber = BER(data, real_data)
		bler = BLER(data, real_data)
		print("step:%s, l_bin:%.4f, l_real:%.4f, l_true:%.4f, d_bin_real:%.4f, d_bin_true:%.4f, d_real_true:%.4f, ber:%.4f, bler:%.4f" % (i, loss_BP_bin, loss_BP_real, loss_true, d_bin_real, d_bin_true, d_real_true, ber, bler))
	sess.run(train, {true_y: noisy_codeword})

	if i%10 == 9:
		sess.run(belief.assign(tf.sign(belief)*(tf.abs(belief)+2.0*tf.nn.relu(tf.abs(belief)-0.3))))

'''
print ("data = ", data)
print ("real = ", real_data)
print ("bin = ", bin_data)
print ("noisy codeword = ", noisy_codeword)
print ("codeword = ", sess.run(y, {x0: data}))
print ("real codeword = ", sess.run(y, {x0: decoded}))
print ("bin codeword = ", sess.run(y, {x0: bin_data}))
'''

