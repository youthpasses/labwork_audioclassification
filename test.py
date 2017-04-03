# coding:utf-8
# @makai
# 16/10/26

import os
# import librosa
import numpy as np

TRAIN_PATH = 'data/traindata/'
TEST_PATH = 'data/testdata/'

'''
def getMFCCs(path):
	filelist = os.listdir(path)
	for i in xrange(0, 50):
		filepath = path + filelist[i]
		y, sr = librosa.load(filepath)
	 	mfcc = librosa.feature.mfcc(y=y, sr=sr)
	 	print i, mfcc[0][0:5]
'''

def saveNative(TIME_DURATION):
	filedir = 'data/data_txt'
	print filedir
	if not os.path.exists(filedir):
		print filedir
	 	os.mkdir(filedir)
	filepath = filedir + '/' + str(TIME_DURATION) + '.txt'
	data = [[[x for x in xrange(0,2)] for y in xrange(0, 3)] for z in xrange(0, 4)]
	data = np.array(data)
	data = np.reshape(data, [data.shape[0], -1])
	# f = open(filepath, 'w+a')
	# f.write(data)
	# f.close()
	f = open(filepath, 'r')
	lines = f.readlines()
	print lines
	f.close()

if __name__ == '__main__':
	saveNative(5)