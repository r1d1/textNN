#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from neuralNet import NeuralNet
from textProcessing import TextProcessing

plt.ion()

proc = TextProcessing()
# Mapping words as image, so need a large input :
wordLength = 20
layerLength = proc.maxChar-proc.minChar
externalSize = layerLength*wordLength
nn = NeuralNet(externalSize, externalSize, externalSize / 2 )

dictFile = open('corncob_lowercase.txt','r')
rawWords = dictFile.read()
dictFile.close()

print len(rawWords), type(rawWords), len(rawWords.split())
dictOfWords = rawWords.split()

alphabet = 'abcdefghijklmnopqrstuvwxyz'
iterNb = 0
errorEvolution=[]
for i in range(100):
#while abs(nn.endError) > 0.01:
	#inputtext = alphabet[np.random.randint(0,len(alphabet)-1)] 
	#answer=""
	# Randomly pick a word from dict :
	inputtext = dictOfWords[np.random.randint(0,len(dictOfWords)-1)]
	print inputtext
	inputvec = [0.0 for i in range(len(nn.inputLayer))]
	counter = 0
	for letter in inputtext:
		inputvec[counter*layerLength + proc.char2val(letter)] = 1.0
		counter += 1
#	print len(inputvec), len(nn.inputLayer)
	# Make network learn from input
#	#for char in inputtext:
#	#	nn.inputData(proc.char2vec(char))
#	#	nn.computeOutput()
#	#	nn.learn()
#	#	answer += proc.vec2char(nn.outputLayer_f.tolist())
	nn.inputData(inputvec)
	nn.computeOutput()
	#nn.learn()
	
	#answer += proc.vec2char(nn.outputLayer_f.tolist())
	# Generate answer (which is way more tricky)
	iterNb = iterNb + 1
	errorEvolution.append(abs(nn.endError))
	#print answer
	#nn.displayNetwork()

#plt.show()
plt.figure()
plt.plot(errorEvolution)
plt.show()
print "Network trained in ", iterNb, "iterations !"

#inputtext = raw_input('Say something: ')
#answer=""
#
## Make network learn from input
##ifor char in inputtext:
##	nn.inputData(proc.char2vec(char))
##	nn.computeOutput()
##	answer += proc.vec2char(nn.outputLayer_f.tolist())
#for letter in inputtext:
#	inputvec[counter*layerLength + proc.char2val(letter)] = 1.0
#	counter += 1
#nn.inputData(inputvec)
#nn.computeOutput()
#
#for letter in range(wordLength):
#	answer += proc.vec2char(nn.outputLayer_f[letter*wordLength:(letter+1)*wordLength].tolist())
## Generate answer (which is way more tricky)
#print answer

while True:
	inputtext = raw_input('Say something: ')
	answer=""
	if inputtext == 'Q' or inputtext == 'q':
		break
	else :
		for letter in inputtext:
			inputvec[counter*layerLength + proc.char2val(letter)] = 1.0
			counter += 1
		nn.inputData(inputvec)
		nn.computeOutput()
	for letter in range(wordLength):
		answer += proc.vec2char(nn.outputLayer_f[letter*wordLength:(letter+1)*wordLength].tolist())
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
