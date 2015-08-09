#!/usr/bin/python

import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNet:
	def __init__(self, inSize=5, outSize=5, hidSize=3):
		self.inputLayer=np.zeros(inSize)
		self.hiddenLayer=np.zeros(hidSize)
		self.outputLayer=np.zeros(outSize)
		self.weights = [np.random.randn(hidSize, inSize), np.random.randn(outSize, hidSize)]
		self.inSize = inSize
		self.outSize = outSize
		self.hidSize = hidSize
	#	print self.weights

	# direct layer value (so a vector of float) at t
	def inputData(self, data):
		# so we can do some processing on input
		if len(data) <= self.inSize:
			self.inputLayer = np.array(data)
	
	# index of fully activated cells (binary array) at t
	def inputPositionalData(self, data):
		# so we can do some processing on input
		for activCell in data:
			print activCell
			if (activCell >= 0) and (activCell < len(self.inputLayer)):
				self.inputLayer[activCell] = 1.0


	def learn(self):
		# Back Propagate :
		# for all layers (inputLayer is desired output)
		error = self.outputLayer * (1.0 - self.outputLayer) * np.power(np.subtract(self.inputLayer, self.outputLayer), 2)
		# error = np.power(np.subtract(self.outputLayer, self.inputLayer), 2)
		print self.weights[1]

	#	print error
		print error.shape, self.sigmoidDeriv(1.0, 500.7), sum(error)
	#	x=np.arange(-100,100)
	#	data1=[self.sigmoid(1.0, val) for val in x]
	#	data2=[self.sigmoidDeriv(1.0, val) for val in x]
	#	plt.plot(data1)
	#	plt.plot(data2)
	#	plt.show()

	def computeOutput(self):
		for neuron in range(self.hidSize):
			self.hiddenLayer[neuron] = self.sigmoid(1.0, np.dot(self.inputLayer, self.weights[0][neuron]))
		#	print len(self.inputLayer), self.weights[0][neuron].shape, len(self.hiddenLayer[neuron].shape)
	#	print self.hiddenLayer
		for neuron in range(self.outSize):
			self.outputLayer[neuron] = self.sigmoid(1.0, np.dot(self.hiddenLayer, self.weights[1][neuron]))
		#	print len(self.weights)
		#	print len(self.outputLayer), self.weights[1][neuron].shape
#		print self.outputLayer, self.outputLayer.tolist().index(max(self.outputLayer))

	def learningRule(self, rule="hebb"):
		if rule == "hebb":
			pass
		elif rule == "lms":
			pass
	
	def sigmoid(self, lamb, val):
		return 1.0 / (1.0 + math.exp( - lamb * val))
	
	def sigmoidDeriv(self, lamb, val):
		# Is also equal to s(x)*(1-s(x))
		return math.exp(-lamb * val) / math.pow((1.0 + math.exp( - lamb * val)),2)
