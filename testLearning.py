#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
#import brewer2mpl
import palettable

from neuralNet import NeuralNet
from textProcessing import TextProcessing
import time
import sys

plt.ion()

inputSize= 10
nn = NeuralNet(inputSize, inputSize, 4)
nn.learningRate = 0.005

# Let's learn patterns of length 5 with a 1-value and two 0.5-values
#dataset =
onePos = [3, 7, 2, 5, 0, 1, 4, 6, 8]
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
print "First errors:"
print "input:"
print inputdata
print "output"
print nn.outputLayer_f
print nn.endError
print meanError
alpha = 0.005

l=0
#while l < 10000:
starttime = time.time()
comptimes= []
print "Training:",
threshold = 0.001
try:
	while meanError > threshold:
		inputdata=dataset[np.random.randint(0,len(dataset)-1)]
		# Make network learn from input
		nn.inputData(inputdata)
		nn.computeOutput()
		nn.learn()
		iterNb = iterNb + 1
		errorEvolution.append(abs(nn.endError))
		meanError = (1 - alpha) * meanError + alpha * abs(nn.endError) 
		# Generate answer (which is way more tricky)
		l=l+1
		comptimes.append(time.time() - starttime)
		sys.stdout.write('\r')
		# the exact output you're looking for:
		sys.stdout.write("%f" % (meanError/threshold))
		sys.stdout.flush()
except KeyboardInterrupt:
	print "Interrupted learning at",l,"iterations"

plt.figure()
nn.displayNetwork()

fig2 = plt.figure()
ax= fig2.add_subplot(2,1,1)
ax.plot(errorEvolution)
ax= fig2.add_subplot(2,1,2)
ax.plot(comptimes)
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
                        #plt.ylim([0,1])
			plt.subplot(2,1,2)
			plt.plot(nn.outputLayer_f)
                        #plt.ylim([0,1])

nn.saveWeights()
