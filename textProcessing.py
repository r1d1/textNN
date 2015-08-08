#!/usr/bin/python

# This script convert a string into its ascii value to input into a NN

class TextProcessing:
	def __init__(self):
		self.rawText = ""
		self.maxChar = 127 # DEL on ASCII table
		self.minChar = 32 # SPACE on ASCII table
		self.output=[]
		self.inputText=""

	def text2Input(self):
		
		self.rawText = raw_input('Say something: ')
		self.rawText = self.rawText.upper()

		printableLine = ['-' for i in range(self.maxChar-self.minChar)]
		for char in self.rawText:
			self.output.append(self.char2val(char))

	def char2val(self, char):
		val = -1
		if (ord(char) < self.maxChar) and (ord(char) >= self.minChar):
			val = (ord(char)-ord(' ')) # First character from ASCII table
		return val
	
	def char2vec(self, char):
		vec = [0 for i in range(self.maxChar-self.minChar)]
		val = self.char2val(char)
		vec[val]=1.0
		return vec
	
	def vec2char(self, vec):
		char = chr(vec.index(max(vec))+self.minChar)
		return char

	def input2Text(self, out):
		for cha in out:
			print cha
			self.inputText += chr(cha+self.minChar)
		self.inputText = self.inputText.lower()
	
	def internalInput2Text(self):
		self.inputText = ""
		for cha in self.output:
			print cha
			self.inputText += chr(cha+self.minChar)
		self.inputText = self.inputText.lower()
			
#for elem in self.output:
#	printedLine = printableLine[:]
#	print elem, ' ',
#	if elem != -1:
#		printedLine[elem] = '#'
#	else:
#		printedLine[elem] = '@'
#	printedLine = ''.join(printedLine)
#	print printedLine

#proc = TextProcessing()
#
#proc.text2Input()
#print proc.rawText
#print proc.output
#print proc.inputText
#
#proc.input2Text()
#print proc.rawText
#print proc.output
#print proc.inputText
