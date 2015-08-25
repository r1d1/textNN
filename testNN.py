#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from neuralNet import NeuralNet
from textProcessing import TextProcessing

plt.ion()

proc = TextProcessing()
nn = NeuralNet(proc.maxChar-proc.minChar, proc.maxChar-proc.minChar, (proc.maxChar-proc.minChar)/2 )

#proc.text2Input()
#print proc.rawText
#print proc.output
#print proc.inputText

#randInput = np.zeros(nn.inSize)
#randLetter = np.random.randint(0, nn.inSize)
#randInput[randLetter] = 1.0
#nn.inputPositionalData([5, 14, 38, 14])
#print randLetter 

alphabet = 'abcdefghijklmnopqrstuvwxyz'
iterNb = 0
#for i in range(100):
while abs(nn.endError) > 0.00001:
	#inputtext = raw_input('Say something: ')
	inputtext = alphabet[np.random.randint(0,len(alphabet)-1)] 
	answer=""
	# Make network learn from input
	for char in inputtext:
		nn.inputData(proc.char2vec(char))
		nn.computeOutput()
		nn.learn()
		#nn.learn()
		answer += proc.vec2char(nn.outputLayer_f.tolist())
	# Generate answer (which is way more tricky)
	iterNb = iterNb + 1
	#print answer
	#nn.displayNetwork()

#plt.show()
print "Network trained in ", iterNb, "iterations !"

inputtext = raw_input('Say something: ')
answer=""

# Make network learn from input
for char in inputtext:
	nn.inputData(proc.char2vec(char))
	nn.computeOutput()
	answer += proc.vec2char(nn.outputLayer_f.tolist())
# Generate answer (which is way more tricky)
print answer

#proc.input2Text([66, 38, 43])
#print proc.rawText
#print proc.output
#print proc.inputText

#print proc.char2val('a')
#print proc.char2val(' ')
#print proc.char2val('*')
#print proc.char2val('Q')
#print proc.char2vec('a')
#print proc.char2vec(' ')
#print proc.char2vec('*')
#print proc.char2vec('Q')
