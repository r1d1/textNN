#!/usr/bin/python

import numpy as np
from neuralNet import NeuralNet
from textProcessing import TextProcessing

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

for i in range(10):
	inputtext = raw_input('Say something: ')
	answer=""
	# Make network learn from input
	for char in inputtext:
		nn.inputData(proc.char2vec(char))
		#print proc.char2val(char), nn.inputLayer
		nn.computeOutput()
		nn.learn()
		#nn.learn()
		answer += proc.vec2char(nn.outputLayer.tolist())
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
