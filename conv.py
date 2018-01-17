import numpy as np
import scipy as sp
import commpy.channelcoding.convcode as cc
import sys

k = 100
n = 2*k+4
SNR = float(sys.argv[1])
sigma = 10**(-0.05*SNR)
testing_size = 6400

generator_matrix = np.array([[05, 07]])
M = np.array([2])
trellis = cc.Trellis(M, generator_matrix)
tb_depth = 5*(M.sum()+1)

def conv_encoder(message_bits, trellis):
	return 2*cc.conv_encode(np.asarray(message_bits), trellis)-1

def conv_decoder(codeword_bits, trellis, tb_depth):
	return cc.viterbi_decode(codeword_bits.astype(float), trellis, tb_depth, decoding_type = 'unquantized')

def rep_encoder(message_bits):
	codeword = np.zeros(2*len(message_bits))
	for i in range(len(message_bits)):
		codeword[2*i] = 2*message_bits[i]-1
		codeword[2*i+1] = 2*message_bits[i]-1
	return codeword

def rep_decoder(codeword_bits):
	decoded = np.zeros((len(codeword_bits)/2))
	for i in range(len(decoded)):
		if codeword_bits[2*i]+codeword_bits[2*i+1] > 0:
			decoded[i] = 1
		else:
			decoded[i] = 0
	return decoded

def BER(data, decoded_data):
	BER = 0.0
	for row in range(len(data)):
		for col in range(len(data[0])):
			if abs(data[row][col] - decoded_data[row][col]) > 0.5:
				BER += 1.0/(len(data)*len(data[0]))
	return BER, np.sqrt(BER*(1-BER)/(len(data)*len(data[0])))

def BLER(data, decoded_data):
	BLER = 0.0
	for row in range(len(data)):
		for col in range(len(data[0])):
			if abs(data[row][col] - decoded_data[row][col]) > 0.5:
				BLER += 1.0/len(data)
				break
	return BLER, np.sqrt(BLER*(1-BLER)/len(data))

data = np.random.randint(2,size=(testing_size,k))
conv_encoded_data = np.zeros((testing_size,n))
rep_encoded_data = np.zeros((testing_size,2*k))

for i in range(testing_size):
	if i%100 == 0:
		print("Encoding number %s" % i)
	conv_encoded_data[i] = conv_encoder(data[i], trellis)
	rep_encoded_data[i] = rep_encoder(data[i])

conv_noisy_data = conv_encoded_data + np.random.normal(0.0,sigma,size=(testing_size,n))
rep_noisy_data = rep_encoded_data + np.random.normal(0.0,sigma,size=(testing_size,2*k))
conv_decoded_data = np.zeros((testing_size,k))
rep_decoded_data = np.zeros((testing_size,k))

for i in range(testing_size):
	if i%100 == 0:
		print("Decoding number %s" % i)
	conv_decoded_data[i] = conv_decoder(conv_noisy_data[i], trellis, tb_depth)[:-int(M)]
	rep_decoded_data[i] = rep_decoder(rep_noisy_data[i])

conv_ber, conv_ber_var = BER(data, conv_decoded_data)
rep_ber, rep_ber_var = BER(data, rep_decoded_data)
conv_bler, conv_bler_var = BLER(data, conv_decoded_data)
rep_bler, rep_bler_var = BLER(data, rep_decoded_data)

print("Blocklength: %s, SNR: %s, Conv BER: %.6f+-%.6f, Rep BER: %.6f+-%.6f , Conv BLER: %.6f+-%.6f, Rep BLER: %.6f+-%.6f " % (k, SNR, conv_ber, conv_ber_var, rep_ber, rep_ber_var, conv_bler, conv_bler_var, rep_bler, rep_bler_var))

