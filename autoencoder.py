import numpy as np
import scipy as sp
import copy
import tensorflow as tf
import commpy.channelcoding.convcode as cc
import sys

k = int(sys.argv[1])
n = 2*k+4
training_size = 200000
testing_size = 10000
sample_size = 2500
train_SNR = 0.0
train_sigma = 10**(-0.05*train_SNR)
test_SNR = float(sys.argv[2])
test_sigma = 10**(-0.05*test_SNR)
learning_rate = 5
momentum = 0.7
steps = 1000
beta = 0.2
prob = 0.1

def BER(data, decoded_data, sample_size):
	BER = 0.0
	sample_rows = np.random.choice(len(data), sample_size, replace=False)
	for row in sample_rows:
		for col in range(len(data[0])):
			if abs(data[row][col] - decoded_data[row][col]) > 0.5:
				BER += 1.0/(sample_size*len(data[0]))
	return BER, np.sqrt(BER*(1-BER)/(sample_size*len(data[0])))

def BLER(data, decoded_data, sample_size):
	BLER = 0.0
	sample_rows = np.random.choice(len(data), sample_size, replace=False)
	for row in sample_rows:
		for col in range(len(data[0])):
			if abs(data[row][col] - decoded_data[row][col]) > 0.5:
				BLER += 1.0/sample_size
				break
	return BLER, np.sqrt(BLER*(1-BLER)/sample_size)

##########################################################################################
#Autoencoder Model Parameters
me = np.linspace(k,n,num=2).astype(int)
md = np.linspace(n,k,num=2).astype(int)
vare, vard = 0.1, 0.1
#Input
x0 = tf.placeholder(tf.float32, [None,k])
noise = tf.placeholder(tf.float32, [None,n])
#Connectivity matrices
Ce_1 = tf.placeholder(tf.float32, [me[0],me[1]])
'''
Ce_2 = tf.placeholder(tf.float32, [me[1],me[2]])
'''

#Layer 1
We_1 = tf.Variable(tf.random_normal([me[0],me[1]], 0.0, vare))
be_1 = tf.Variable(tf.random_normal([me[1]], 0.0, vare))
'''
#Layer 2
We_2 = tf.Variable(tf.random_normal([me[1],me[2]], 0.0, vare))
be_2 = tf.Variable(tf.random_normal([me[2]], 0.0, vare))
'''

#Layer 1
Wd_1 = tf.Variable(tf.random_normal([md[0],md[1]],0.0,vard))
bd_1 = tf.Variable(tf.random_normal([md[1]],0.0,vard))
'''
#Layer 2
Wd_2 = tf.Variable(tf.random_normal([md[1],md[2]],0.0,vard))
bd_2 = tf.Variable(tf.random_normal([md[2]],0.0,vard))
'''

#Encoder Model
#Layer 1
y1 = 2*tf.nn.sigmoid(tf.matmul(x0, tf.multiply(Ce_1, We_1)) + be_1)-1
x1 = np.sqrt(me[1])*tf.nn.l2_normalize(y1, dim=1)
'''
#Layer 2
y2 = 2*tf.nn.sigmoid(tf.matmul(x1, tf.multiply(Ce_2, We_2)) + be_2)-1
x2 = np.sqrt(me[2])*tf.nn.l2_normalize(y2, dim=1)
'''
z0 = x1 + noise

#Decoder Model
#Layer 1
z1 = 2*tf.nn.sigmoid(tf.matmul(z0, Wd_1) + bd_1)-1
'''
#Layer 2
z2 = 2*tf.nn.sigmoid(tf.matmul(z1, Wd_2) + bd_2)-1
'''
z = 0.5*(1+z1)

#Prepare Data
training_data = np.random.randint(2,size=[training_size,k])
testing_data = np.random.randint(2,size=[testing_size,k])
training_noise = np.random.normal(0.0,train_sigma,size=[training_size,n])
testing_noise = np.random.normal(0.0,test_sigma,size=[testing_size,n])
C_1 = np.random.binomial(1,prob,size=[me[0],me[1]])
'''
C_2 = np.random.binomial(1,prob,size=[me[1],me[2]])
'''

#Loss Function
loss_l2 = tf.reduce_mean(tf.square(z-x0))
loss_crossentropy = tf.reduce_mean(-tf.log(tf.clip_by_value(tf.abs(1.0-z-x0),1e-15,1.0)))
#regularizer = tf.reduce_mean(tf.abs(We_1))+tf.reduce_mean(tf.abs(be_1))+tf.reduce_mean(tf.abs(We_2))+tf.reduce_mean(tf.abs(be_2))
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train = optimizer.minimize(loss_crossentropy)

#Training and Testing Data
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(steps):
	if i%10 == 0:
#		training_loss = sess.run(loss, {x0: training_data, noise: training_noise})
		testing_loss = sess.run(loss_l2, {x0: testing_data, noise: testing_noise, Ce_1: C_1})
#		training_BER, training_BER_var = BER(training_data, sess.run(z, {x0: training_data, noise: training_noise}), sample_size)
		testing_BER, testing_BER_var = BER(testing_data, sess.run(z, {x0: testing_data, noise: testing_noise, Ce_1: C_1}), sample_size)
#		training_BLER, training_BLER_var = BLER(training_data, sess.run(z, {x0: training_data, noise: training_noise}), sample_size)
		testing_BLER, testing_BLER_var = BLER(testing_data, sess.run(z, {x0: testing_data, noise: testing_noise, Ce_1: C_1}), sample_size)
		print("step:%s, loss:%.6f, BER:%.6f +- %.6f, BLER:%.4f +- %.6f" % (i, testing_loss, testing_BER, testing_BER_var, testing_BLER, testing_BLER_var))
	sess.run(train, {x0: training_data, noise: training_noise, Ce_1: C_1})

