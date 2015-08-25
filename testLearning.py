#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl

from neuralNet import NeuralNet
from textProcessing import TextProcessing

plt.ion()

inputSize= 10
nn = NeuralNet(inputSize, inputSize, 4)

# Let's learn patterns of length 5 with a 1-value and two 0.5-values
#dataset =
onePos = [3, 7, 2, 5, 0]
patterns = [[0.0 if elem != onePos[samp] else 1.0 for elem in range(inputSize)] for samp in range(len(onePos))] 

print patterns

datasetSize=500
dataset=[]
# generate dataset of size 500
for d in range(datasetSize):
	pat = np.random.randint(0,len(patterns))
	sample = [patterns[pat][elem]+0.25*np.random.rand() for elem in range(inputSize)]
	dataset.append(sample)

#print "dataset"
#print dataset
# brewer2mpl.get_map args: set name set type number of colors
#bmap = brewer2mpl.get_map('Paired', 'qualitative', datasetSize)
#colors = bmap.mpl_colors
#samp = 0
#for data in dataset:
#	plt.subplot(5,2,samp)
#	plt.plot(data, color=colors[samp])
#	samp += 1

#nn.displayNetwork()
errorEvolution=[]
# First example :
inputdata=dataset[np.random.randint(0,len(dataset)-1)]
nn.inputData(inputdata)
nn.computeOutput()
nn.learn()
iterNb = 1
errorEvolution.append(abs(nn.endError))
meanError = abs(nn.endError)
alpha = 0.1

while meanError > 0.01:
	inputdata=dataset[np.random.randint(0,len(dataset)-1)]
#	# Make network learn from input
	nn.inputData(inputdata)
	nn.computeOutput()
	nn.learn()
	iterNb = iterNb + 1
	errorEvolution.append(abs(nn.endError))
	meanError = (1 - alpha) * meanError + alpha * abs(nn.endError) 
#	# Generate answer (which is way more tricky)

plt.figure()
nn.displayNetwork()

plt.figure()
plt.plot(errorEvolution)
plt.show()
print "Network trained in ", iterNb, "iterations !"
raw_input('Press Enter to continue. ')

print "Testing phase"

testDatasetSize=3 * datasetSize
testDataset=[]
# generate dataset of size 500
for d in range(testDatasetSize):
	pat = np.random.randint(0,len(patterns))
	sample = [patterns[pat][elem]+0.25*np.random.rand() for elem in range(inputSize)]
	testDataset.append(sample)

finalError=0
for expl in testDataset:
	nn.inputData(expl)
	nn.computeOutput()
	finalError += abs(expl - nn.outputLayer_f)

# normalising by neuron and by sample nb
finalError = sum(finalError) / (inputSize * testDatasetSize)
print "Final error", finalError

count = 0
for pat in patterns:
	print count, pat
	count += 1

while True:
	ptT = raw_input('Which pattern to test ? (Or Q to quit) : ')
	print type(ptT)
	if ptT == 'Q' or ptT == 'q':
		break
	else :
		if ptT < str(len(patterns)):
			pat = int(ptT)
			#print patterns[pat]
			sample = [patterns[pat][elem]+0.25*np.random.rand() for elem in range(inputSize)]
			print sample
			nn.inputData(sample)
			nn.computeOutput()
			print nn.outputLayer_f
			plt.clf()	
			plt.subplot(2,1,1)
			plt.plot(nn.inputLayer)
			plt.subplot(2,1,2)
			plt.plot(nn.outputLayer_f)

