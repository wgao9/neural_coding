import numpy as np
import scipy as sp
import copy
import tensorflow as tf
import commpy.channelcoding.convcode as cc

k = 100
n = 204
training_size = 200000
testing_size = 1000
SNR = -3.0
sigma = 10**(-0.05*SNR)
learning_rate = 10
steps = 1000

def conv_encoder(message_bits):
	generator_matrix = np.array([[05, 07]])
	M = np.array([2])
	trellis = cc.Trellis(M, generator_matrix)
	return 2*cc.conv_encode(np.asarray((1+message_bits)/2), trellis)-1

def conv_decoder(coded_bits):
	generator_matrix = np.array([[05, 07]])
	M = np.array([2])
	trellis = cc.Trellis(M, generator_matrix)
	tb_depth = 5*(M.sum()+1)
	return 2*cc.viterbi_decode(((coded_bits+1)/2).astype(float), trellis, tb_depth)-1

def BER(data, decoded_data):
	BER = 0.0
	for row in range(len(data)):
		for col in range(len(data[0])):
			if abs(data[row][col] - decoded_data[row][col]) > 0.5:
				BER += 1.0
	return BER/(len(data)*len(data[0]))

##########################################################################################
#Encoder Model Parameters
m = np.linspace(k,n,num=2).astype(int)
#Input
x0 = tf.placeholder(tf.float32, [None,k])
#Layer 1
We_1 = tf.get_variable("We_1", shape=[m[0],m[1]])
be_1 = tf.get_variable("be_1", shape=[m[1]])
'''
#Layer 2
We_2 = tf.get_variable("We_2", shape=[m[1],m[2]])
be_2 = tf.get_variable("be_2", shape=[m[2]])
#Layer 3
We_3 = tf.get_variable("We_3", shape=[m[2],m[3]])
be_3 = tf.get_variable("be_3", shape=[m[3]])
#Layer 4
We_4 = tf.get_variable("We_4", shape=[m[3],m[4]])
be_4 = tf.get_variable("be_4", shape=[m[4]])
'''

#Model
#Layer 1
y1 = 2*tf.nn.sigmoid(tf.matmul(x0, We_1) + be_1)-1
x1 = np.sqrt(m[1])*tf.nn.l2_normalize(y1, dim=1)
''''
#Layer 2
y2 = 2*tf.nn.sigmoid(tf.matmul(x1, We_2) + be_2)-1
x2 = np.sqrt(m[2])*tf.nn.l2_normalize(y2, dim=1)
#Layer 3
y3 = 2*tf.nn.sigmoid(tf.matmul(x2, We_3) + be_3)-1
x3 = np.sqrt(m[3])*tf.nn.l2_normalize(y3, dim=1)
#Layer 4
y4 = 2*tf.nn.sigmoid(tf.matmul(x3, We_4) + be_4)-1
x4 = np.sqrt(m[4])*tf.nn.l2_normalize(y4, dim=1)
'''
y = x1

saver = tf.train.Saver()
sess = tf.Session() 
saver.restore(sess, "/home/wgao9/encoder/tmp/model_k=100_n=204_1layer.ckpt")
'''
#Prepare Data
training_data = np.random.randint(2,size=(training_size,k))
testing_data = np.random.randint(2,size=(testing_size,k))
training_codeword = sess.run(y, {x0:training_data})
testing_codeword = sess.run(y, {x0:testing_data})
training_noisy = training_codeword + np.random.normal(0.0,sigma,size=[training_size,n])
testing_noisy = testing_codeword + np.random.normal(0.0,sigma,size=[testing_size,n])

#######################################################################################
#######################################################################################
m = np.linspace(n,k,num=2).astype(int)
var = 0.1
#Decoding Model Parameters
z0 = tf.placeholder(tf.float32,[None,n])
true_z = tf.placeholder(tf.float32,[None,k])
#Layer 1
Wd_1 = tf.Variable(tf.random_normal([m[0],m[1]],0.0,var))
bd_1 = tf.Variable(tf.random_normal([m[1]],0.0,var))
'''
'''
#Layer 2
Wd_2 = tf.Variable(tf.random_normal([m[1],m[2]],0.0,var))
bd_2 = tf.Variable(tf.random_normal([m[2]],0.0,var))
#Layer 3
Wd_3 = tf.Variable(tf.random_normal([m[2],m[3]],0.0,var))
bd_3 = tf.Variable(tf.random_normal([m[3]],0.0,var))
#Layer 4
Wd_4 = tf.Variable(tf.random_normal([m[3],m[4]],0.0,var))
bd_4 = tf.Variable(tf.random_normal([m[4]],0.0,var))
#Layer 5
Wd_5 = tf.Variable(tf.random_normal([m[4],m[5]],0.0,var))
bd_5 = tf.Variable(tf.random_normal([m[5]],0.0,var))
#Layer 6
Wd_6 = tf.Variable(tf.random_normal([m[5],m[6]],0.0,var))
bd_6 = tf.Variable(tf.random_normal([m[6]],0.0,var))
#Layer 7
Wd_7 = tf.Variable(tf.random_normal([m[6],m[7]],0.0,var))
bd_7 = tf.Variable(tf.random_normal([m[7]],0.0,var))
#Layer 8
Wd_8 = tf.Variable(tf.random_normal([m[7],m[8]],0.0,var))
bd_8 = tf.Variable(tf.random_normal([m[8]],0.0,var))
'''

'''
#Decoder Model
#Layer 1
z1 = 2*tf.nn.sigmoid(tf.matmul(z0, Wd_1) + bd_1)-1
'''
'''
#Layer 2
z2 = 2*tf.nn.sigmoid(tf.matmul(z1, Wd_2) + bd_2)-1
#Layer 3
z3 = 2*tf.nn.sigmoid(tf.matmul(z2, Wd_3) + bd_3)-1
#Layer 4
z4 = 2*tf.nn.sigmoid(tf.matmul(z3, Wd_4) + bd_4)-1
#Layer 5
z5 = 2*tf.nn.sigmoid(tf.matmul(z4, Wd_5) + bd_5)-1
#Layer 6
z6 = 2*tf.nn.sigmoid(tf.matmul(z5, Wd_6) + bd_6)-1
#Layer 7
z7 = 2*tf.nn.sigmoid(tf.matmul(z6, Wd_7) + bd_7)-1
#Layer 8
z8 = 2*tf.nn.sigmoid(tf.matmul(z7, Wd_8) + bd_8)-1
'''
'''
z = 0.5*(1+z1)

#Loss Function
loss = tf.reduce_mean(-tf.log(tf.abs(1.0-z-true_z)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

#Training and Testing Data
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(steps):
	if i%100 == 0:
		training_loss = sess.run(loss, {z0: training_noisy, true_z: training_data})
		testing_loss = sess.run(loss, {z0: testing_noisy, true_z: testing_data})
		training_BER = BER(training_data, sess.run(z, {z0: training_noisy}))
		testing_BER = BER(testing_data, sess.run(z, {z0: testing_noisy}))
		print("step:%s, training loss:%s, testing loss:%s, training BER:%s, testing BER:%s" % (i, training_loss, testing_loss, training_BER, testing_BER))
	sess.run(train, {z0: training_noisy, true_z: training_data})

'''
'''
data = 2*np.random.randint(2,size=(testing_size,k))-1
error_rate = 0.0
for i in range(testing_size):
	coded_bits = conv_encoder(data[i])
	noisy_bits = coded_bits + np.random.normal(0.0,sigma,n)
	decoded_bits = conv_decoder(noisy_bits)
	if i%50 == 0:
		print np.inner(data[i],decoded_bits[:k])
		print np.inner(coded_bits,noisy_bits)
	for ind in range(k):
		if data[i][ind] != decoded_bits[ind]:
			error_rate += 1.0/(testing_size*k)
			#break

print(error_rate)


neural_coded_data = neural_encoder(data,k,n)
#print(neural_coded_data.eval())
'''

