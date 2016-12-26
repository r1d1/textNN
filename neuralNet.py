#!/usr/bin/python

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import datetime

class NeuralNet:
	def __init__(self, inSize=5, outSize=5, hidSize=3):
		self.inputLayer=np.zeros(inSize)
		# *_r is raw data (dot product of input and weights, *_f is after function is applied
		self.hiddenLayer_r=np.zeros(hidSize)
		self.hiddenLayer_f=np.zeros(hidSize)
		self.outputLayer_r=np.zeros(outSize)
		self.outputLayer_f=np.zeros(outSize)
		#self.weights = [np.random.randn(hidSize, inSize), np.random.randn(outSize, hidSize)]
		self.weights = [np.random.rand(hidSize, inSize), np.random.rand(outSize, hidSize)]
		self.inSize = inSize
		self.outSize = outSize
		self.hidSize = hidSize
		self.learningRate = 0.1
		self.endError = sys.float_info.max

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
		error = self.outputLayer_f * (1.0 - self.outputLayer_f) * np.subtract(self.inputLayer, self.outputLayer_f)
		#print "error T",sum(error)
		# Must be hidden layer size :
		errorProp = np.array([ sum( error * self.weights[1][:, neuron] ) for neuron in np.arange(self.hidSize)])
		nexterror = self.hiddenLayer_f * (1.0 - self.hiddenLayer_f) * errorProp
	#	print type(error), type(nexterror), len(nexterror), len(error), errorProp.shape
		#plt.plot(error)
		#plt.plot(nexterror)
		#plt.draw()

		#print len(self.weights[1][neuron]), self.learningRate, "error", len(error), "hid layer", len(self.hiddenLayer_f)
		#print type(nexterror[3]), type(error[3]), type(self.inputLayer[5])
		variation = self.weights[1]
		for outN in range(self.outSize):
			for inN in range(self.hidSize):
				self.weights[1][outN][inN] = self.weights[1][outN][inN] + self.learningRate * error[outN] * self.hiddenLayer_f[inN]

		variation = variation - self.weights[1]
		variation = sum(sum(variation))
		#print variation

		for hidN in range(self.hidSize):
			for inN in range(self.inSize):
				self.weights[0][hidN][inN] = self.weights[0][hidN][inN] + self.learningRate * nexterror[hidN] * self.inputLayer[inN]
		self.endError = sum(error)
		#print "remaining error",sum(error), sum(nexterror)

	def computeOutput(self):
		for neuron in range(self.hidSize):
			self.hiddenLayer_r[neuron] = np.dot(self.inputLayer, self.weights[0][neuron])
			#print self.hiddenLayer_r[neuron]
			self.hiddenLayer_f[neuron] = self.sigmoid(1.0, self.hiddenLayer_r[neuron])
		for neuron in range(self.outSize):
			self.outputLayer_r[neuron] = np.dot(self.hiddenLayer_f, self.weights[1][neuron])
			self.outputLayer_f[neuron] = self.sigmoid(1.0, self.outputLayer_r[neuron])

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
	
	def displayNetwork(self):
		plt.clf()	
		plt.subplot(4,2,1)
		plt.imshow(self.weights[0], interpolation='none')
		plt.colorbar()
		plt.subplot(4,2,2)
		plt.imshow(self.weights[1].T, interpolation='none')
		plt.colorbar()
		plt.subplot(4,2,3)
		plt.plot(self.inputLayer)
		plt.subplot(4,2,4)
		#ttd1 = chr(self.inputLayer.tolist().index(max(self.inputLayer))+32)
		#plt.text(0.5,0.25, ttd1)
		#ttd2 = chr(self.outputLayer_f.tolist().index(max(self.outputLayer_f))+32)
		#plt.text(0.5,0.75, ttd2)
		plt.subplot(4,2,5)
		plt.plot(self.hiddenLayer_r)
		plt.subplot(4,2,6)
		plt.plot(self.hiddenLayer_f)
		plt.subplot(4,2,7)
		plt.plot(self.outputLayer_r)
		plt.subplot(4,2,8)
		plt.plot(self.outputLayer_f)
		plt.draw()

	def saveWeights(self):
		now = datetime.datetime.now()
		saveFile = "weightsSaved_"+str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)
		np.save(saveFile, self.weights)
	
	def loadWeights(self, loadFile):
		self.weights = np.save(loadFile)
