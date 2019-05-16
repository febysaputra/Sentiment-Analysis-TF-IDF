import re
from collections import defaultdict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pprint
import json
import csv
import os
import pandas as pd

import numpy as np
from neupy import algorithms, utils

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import minmax_scale

from nimblenet.activation_functions import sigmoid_function
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.data_structures import Instance
from nimblenet.neuralnet import NeuralNet

from nimblenet.learning_algorithms import backpropagation
from nimblenet.learning_algorithms import backpropagation_classical_momentum
from nimblenet.learning_algorithms import backpropagation_nesterov_momentum
from nimblenet.learning_algorithms import resilient_backpropagation

from tesaurus.tesaurus import *

import string
import itertools

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def readTxt (srcfile): # list stopword
	result = list()
	with open(srcfile) as f:
		for line in f:
			line = line.replace("\n", "")
			result.append(line)
		f.close()
	return result

def readCSVPembakuanData (srcfile): # buat pembakuan data
	kamusPembakuanData = dict()
	csvfile = open(srcfile, 'r')
	reader = csv.DictReader(csvfile)
	for row in reader:
		kamusPembakuanData[row['singkat']] = row['hasil']
	csvfile.close()
	return kamusPembakuanData

def readCSV (srcfile, keywords): # buat pembakuan data
	text = list()
	csvfile = open(srcfile, 'r')
	reader = csv.DictReader(csvfile)
	for row in reader:
		text.append(row[keywords])
	csvfile.close()
	return text

def cleanData (text):
    result = re.sub(r'((\w+:\/{2}[\d\@\w-]+|www\.[\w-]+)+(\.[\d\w-]+)*(?:(?:\/[^\s\/]*))*|([\w]+-\w+@[\w-]+|\w+@[\w-]+)+(\.[\d\w-]+)*(?:(?:\/[^\s\/]*))*)', ' ', text)
    result = re.sub("(@[A-Za-z0-9.,\/#!$%\^&\*;:{}=\-_`~()]+)"," ",result)
    result = re.sub("(#[A-Za-z0-9.,\/#!$%\^&\*;:{}=\-_`~()]+)"," ",result)
    result = re.sub("(?:[^]|[?""''!:;,.()&A-Za-z])+(?=)", " ", result) #pengambilan huruf saja
    result = result.translate(str.maketrans('', '', string.punctuation)) #penghapusan tanda baca
    result = result.replace("RT", " ")
    result = result.lower()
    result = stemmer.stem(result) #stemming
    return result

def tokenisasiAllData(text, listStopWords, kamusPembakuanData, negationList, kbbi):
	term = list()
	for teks in text:
		teks = cleanData(teks) #clean Data
		listKata = teks.split() #tokenisasi per row

		for i in range(len(listKata)):
			if listKata[i] in kamusPembakuanData.keys():
				listKata[i] = kamusPembakuanData[listKata[i]] #pembakuan data
				if len(listKata[i].split()) > 1:
					temp = listKata[i].split()
					for l in range(len(temp)):
						if temp[l] in negationList:
							if(l+1) < len(temp):
								if len(getAntonim(temp[l+1])) != 0 and getAntonim(temp[l+1])[0] not in listStopWords:
									if getAntonim(temp[l+1])[0] not in term:
										term.append(getAntonim(temp[l+1])[0])
										l+=1
						elif temp[l] not in listStopWords:
							if temp[l] not in term and temp[l] not in getSinonim(temp[l]):
								term.append(temp[l])				
				elif listKata[i] in negationList:
					if (i+1) < len(listKata):
						if len(getAntonim(listKata[i+1])) != 0 and getAntonim(listKata[i+1])[0] not in listStopWords:
							if getAntonim(listKata[i+1])[0] not in term:
								term.append(getAntonim(listKata[i+1])[0])
								i+=1	
				elif listKata[i] not in listStopWords: #penghapusan stopwords
					if listKata[i] not in term and listKata[i] not in getSinonim(listKata[i]):
						term.append(listKata[i])
			else:
				if listKata[i] not in kbbi:
					if (''.join(ch for ch, _ in itertools.groupby(listKata[i]))) in kbbi:
						listKata[i] = ''.join(ch for ch, _ in itertools.groupby(listKata[i]))
				if listKata[i] in negationList:
					if (i+1) < len(listKata):
						if len(getAntonim(listKata[i+1])) != 0 and getAntonim(listKata[i+1])[0] not in listStopWords:
							if getAntonim(listKata[i+1])[0] not in term:
								term.append(getAntonim(listKata[i+1])[0])
								i+=1	
				elif listKata[i] not in listStopWords: #penghapusan stopwords
					if listKata[i] not in term and listKata[i] not in getSinonim(listKata[i]):
						term.append(listKata[i])
	return term

def perhitunganTF(text, term, listStopWords, kamusPembakuanData, negationList, kbbi):
	wordSet = set(term)
	wordDict = list() #ini nyimpen tf disetiap row
	for i in range(0, len(text)):
		wordDict.append(dict.fromkeys(wordSet, 0))

		text[i] = cleanData(text[i])
		listKata = text[i].split() #tokenisasi per row
		for j in range(len(listKata)):
			if listKata[j] in kamusPembakuanData.keys():
				listKata[j] = kamusPembakuanData[listKata[j]] #pembakuan data
				if len(listKata[j].split()) > 1:
					temp = listKata[j].split()
					for l in range(len(temp)):
						if temp[l] in negationList:
							if(l+1) < len(temp):
								if len(getAntonim(temp[l+1])) != 0 and getAntonim(temp[l+1])[0] not in listStopWords:
									if getAntonim(temp[l+1])[0] in wordDict[i].keys():
										wordDict[i][getAntonim(temp[l+1])[0]] += 1
										l+=1
						elif temp[l] not in listStopWords:
							if temp[l] not in getSinonim(temp[l]) and temp[l] in wordDict[i].keys():
								wordDict[i][temp[l]]+=1
							else:
								for k in getSinonim(temp[l]):
									if k in wordDict[i].keys():
										wordDict[i][k]+=1
										break
				elif listKata[j] in negationList:
					if (j+1) < len(listKata):
						if len(getAntonim(listKata[j+1])) != 0 and getAntonim(listKata[j+1])[0] not in listStopWords:
							if getAntonim(listKata[j+1])[0] in wordDict[i].keys():
								wordDict[i][getAntonim(listKata[j+1])[0]]+=1
								j+=1
				elif listKata[j] not in listStopWords: #penghapusan stopwords
					if listKata[j] not in getSinonim(listKata[j]) and listKata[j] in wordDict[i].keys():
						wordDict[i][listKata[j]]+=1 #hitung tf per kata
					else:
						for k in getSinonim(listKata[j]):
							if k in wordDict[i].keys():
								wordDict[i][k]+=1 #hitung tf per kata
								break
			else:
				if listKata[j] not in kbbi:
					if (''.join(ch for ch, _ in itertools.groupby(listKata[j]))) in kbbi:
						listKata[j] = ''.join(ch for ch, _ in itertools.groupby(listKata[j]))
				if listKata[j] in negationList:
					if (j+1) < len(listKata):
						if len(getAntonim(listKata[j+1])) != 0 and getAntonim(listKata[j+1])[0] not in listStopWords:
							if getAntonim(listKata[j+1])[0] in wordDict[i].keys():
								wordDict[i][getAntonim(listKata[j+1])[0]]+=1
								j+=1
				elif listKata[j] not in listStopWords: #penghapusan stopwords
					if listKata[j] not in getSinonim(listKata[j]) and listKata[j] in wordDict[i].keys():
						wordDict[i][listKata[j]]+=1 #hitung tf per kata
					else:
						for k in getSinonim(listKata[j]):
							if k in wordDict[i].keys():
								wordDict[i][k]+=1 #hitung tf per kata
								break
	for i in range(len(text)):
		res = list()
		for j in range(len(term)):
			if wordDict[i][term[j]] > 0:
				res.append(term[j])
		if len(res) > 0:
			for word in wordDict[i].keys():
				wordDict[i][word] /= float(len(res)) 
	return wordDict

def perhitunganIDF(docList): #doclist -> wordDict
    import math
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict

def perhitunganTFIDF(wordDict, idfs):
	for i in range(0, len(wordDict)):
		for word, val in wordDict[i].items():
			if(val > 0):
				# val = 1
				wordDict[i][word] = val*idfs[word]
		#wordDict[i]['labeledClass'] = kelas[i] #add labeled class kalo mau diappend ke tfidf
	return wordDict

def vektorisasiData(wordDict):
	dataVektor = list()
	for i in range(0, len(wordDict)):
		dictlist = list()
		for key, value in wordDict[i].items():
		    dictlist.append(value)
		#dictlist = minmax_scale(dictlist) #normalize
		dataVektor.append(dictlist)
	return dataVektor

def maxMinAtr(wordDict):
	maks = dict()
	mins = dict()
	for word in wordDict[0].keys():
		tempMax = 0
		tempMin = 999999
		for i in range(len(wordDict)):
			if wordDict[i][word] > tempMax:
				tempMax = wordDict[i][word]
			if wordDict[i][word] < tempMin:
				tempMin = wordDict[i][word]
		maks[word] = tempMax
		mins[word] = tempMin
	return maks, mins

def normalisasiDataMinMax(wordDict, maks, mins): #min - max
	for word in wordDict[0].keys():
		for i in range(len(wordDict)):
			wordDict[i][word] = (wordDict[i][word]-mins[word])/(maks[word]-mins[word])
	return wordDict

#main
stopWordsFile = 'id.stopwords.02.01.2016.txt'
pembakuanDataFile = 'key_norm.csv'
textFile = 'dataModelWithoutDuplicate.csv'
newDataFile = 'pascaDebatAkhir.csv'
negationFile = 'negation_list.txt'
kbbiFile = 'kata_dasar_kbbi.csv'

if __name__ == '__main__':
	stopWords = readTxt(stopWordsFile)
	negationList = readTxt(negationFile)
	pembakuanData = readCSVPembakuanData(pembakuanDataFile)
	text = readCSV(textFile, "text")
	newData = readCSV(newDataFile, "text")
	kelas = readCSV(textFile, "kelas_manual")
	kbbi = readCSV(kbbiFile, "kata")

	term = tokenisasiAllData(text,stopWords,pembakuanData, negationList, kbbi)
	
	wordDict = perhitunganTF(text, term, stopWords, pembakuanData, negationList, kbbi)
	idfs = perhitunganIDF(wordDict)
	wordDict = perhitunganTFIDF(wordDict, idfs)
	
	maksMins = maxMinAtr(wordDict)
	wordDict = normalisasiDataMinMax(wordDict, maksMins[0], maksMins[1])

	dataVektor = vektorisasiData(wordDict)

	# termNewData = tokenisasiAllData(newData,stopWords,pembakuanData, negationList, kbbi)
	
	wordDictNewData = perhitunganTF(newData, term, stopWords, pembakuanData, negationList, kbbi)
	idfsNewData = perhitunganIDF(wordDictNewData)
	wordDictNewData = perhitunganTFIDF(wordDictNewData, idfsNewData)

	maksMinsNew = maxMinAtr(wordDictNewData)
	wordDictNewData = normalisasiDataMinMax(wordDictNewData, maksMinsNew[0], maksMinsNew[1])

	dataVektorNewData = vektorisasiData(wordDictNewData)

	#buat save file si dataframe
	# df = pd.DataFrame(wordDict)
	# df.to_csv(r'E:/S2/Kuliah/Data Mining/projek akhir/tfidf_BMNB.csv') #ganti aja directorynya.
	
	#mengkodekan string kelas ke data float
	for i in range(0,len(kelas)):
		if kelas[i] == "Positif":
			kelas[i] = float(0)
		elif kelas[i] == "Negatif":
			kelas[i] = float(1)
		elif kelas[i] == "Netral":
			kelas[i] = float(2)

	#create data train and test
	X = np.array(dataVektor)

	y = np.array(kelas)

	XNewData = np.array(dataVektorNewData)

	utils.reproducible(1234) #utk seed

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	# SVM
	# clf = SVC(gamma='auto', decision_function_shape='ovr')
	# clf.fit(X_train, y_train)
	# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
 #    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
 #    max_iter=-1, probability=False, random_state=None, shrinking=True,
 #    tol=0.001, verbose=False)
	# y_pred = clf.predict(X_test)

	# print("SVM\n")
	# print(confusion_matrix(y_test, y_pred))
	# print(accuracy_score(y_test, y_pred))
	# print("\n######")

	#Multinomial Naive Bayes
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
	y_pred = clf.predict(XNewData)
	# print(y_pred)
	filePred = open(r"E:/S2/Kuliah/Data Mining/projek akhir/ypred_CI.txt","w+")
	predFile = list()
	for value in y_pred:
		predFile.append(str(value))
	filePred.writelines(predFile)
	# filePred = open(r"E:/S2/Kuliah/Data Mining/projek akhir/ypred_Test.txt","w+")
	# filePred.write(str(y_pred))

	# print("BMNB\n")
	# print(confusion_matrix(y_test, y_pred))
	# print(accuracy_score(y_test, y_pred))
	# print("\n######")

	# ##LVQ python3
	# lvqnet = algorithms.LVQ3(n_inputs=len(wordDict[0]), n_classes=2)
	# lvqnet.train(X_train, y_train, epochs=50)
	# y_pred = lvqnet.predict(X_test)

	# print("LVQ\n")
	# print(confusion_matrix(y_test, y_pred))
	# print(accuracy_score(y_test, y_pred))
	# print("\n######")
	# ###

	# ##Backprop python2
	#setting networks
	
	# dataset_train = list()
	# for i in range (0,len(X_train)):
	# 	y_train_backprop = list()
	# 	if kelas[i] == 0:
	# 		y_train_backprop = [1,0,0]
	# 	elif kelas[i] == 1:
	# 		y_train_backprop = [0,1,0]
	# 	elif kelas[i] == 2:
	# 		y_train_backprop = [0,0,1]
	# 	dataset_train.append(Instance(X_train[i],y_train_backprop))

	# dataset_test = list()
	# for i in range (0,len(X_test)):
	# 	y_test_backprop = list()
	# 	if kelas[i] == 0:
	# 		y_test_backprop = [1,0,0]
	# 	elif kelas[i] == 1:
	# 		y_test_backprop = [0,1,0]
	# 	elif kelas[i] == 2:
	# 		y_test_backprop = [0,0,1]
	# 	dataset_test.append(Instance(X_test[i],y_test_backprop))

	# n_node_hidden = int((len(wordDict)+3) * (2.00/3))

	# settings = {
	#     "n_inputs" : len(wordDict[0]),
	#     "layers"   : [  (n_node_hidden, sigmoid_function), (3, sigmoid_function) ] #hidden layer , output layer
	# }

	# network = NeuralNet(settings)
	# training_set = dataset_train
	# test_set = dataset_test
	# cost_function = cross_entropy_cost

	# #backprop
	# print("\nTrain Backprop\n")
	# training
	# backpropagation(
	#     # Required parameters
	#     network,                     # the neural network instance to train
	#     training_set,                # the training dataset
	#     test_set,                    # the test dataset
	#     cost_function,               # the cost function to optimize
	#     save_trained_network = True,
	#     print_rate = 1, #ngprint setiap n epoch
	#     max_iterations = 100
 #    )
	# print("\n##########")

	# #open network from training
	# network = NeuralNet.load_network_from_file("network0.pkl")

	# testData = list()
	# for i in range (0,len(X_test)):
	# 	testData.append(Instance(X_test[i]))

	# predict = network.predict(testData)
	# test_pred = list()

	# for i in range(0,len(predict)):
	# 	maxposition = np.argmax(predict[i])
	# 	if maxposition == 0:
	# 		test_pred.append(0)
	# 	elif maxposition == 1:
	# 		test_pred.append(1)
	# 	elif maxposition == 2:
	# 		test_pred.append(2)

	# print("\nBackprop\n")
	# print(confusion_matrix(y_test, test_pred))
	# print(accuracy_score(y_test, test_pred))
	# print("\n######")