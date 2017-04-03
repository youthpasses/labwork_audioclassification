# coding:utf-8
# @makai
# 16/12/08

import librosa
import os
import numpy as np
import random

TIME_DURATION = 3
ROOT_DIR = 'data/10classes/'
# ROOT_DIR = 'data/20classes/'
n_classes = 10

def getLMS(filepath):
	y, sr = librosa.load(filepath)
	ms = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, fmax=8000)
	lms = -librosa.logamplitude(ms, ref_power=np.max)
	lms = np.delete(lms, len(lms[0]) - 1, axis=1)
	lms = np.delete(lms, 0, axis=1)
	lms = np.transpose(lms)
	#print lms.shape
	return lms


def saveLMSFeatures(txtdir, datatype):
	DATA_PATH = ROOT_DIR + str(TIME_DURATION) + '/' + datatype + '/'
	filelist = os.listdir(DATA_PATH)
	f_data = open(txtdir + 'lms_' + datatype + '_data.txt', 'w+a')
	f_label = open(txtdir + 'lms_' + datatype + '_label.txt', 'w+a')
	i = 0
	for i, filename in enumerate(filelist):
		filepath = DATA_PATH + filename
		lms = np.array(getLMS(filepath))
		c = filename.split('_')[0]
		lms = np.reshape(lms, [-1])
		lms = lms.astype(np.str)
		lms = ' '.join(lms) + '\n'
		f_data.write(lms)
		f_label.write(c + '\n')
		if i % 100 == 0:
			print i, ': getLMS: ' + filepath
	f_data.close()
	f_label.close()

def getLMSFeatures():
	txtdir = ROOT_DIR + 'data_txt/' + str(TIME_DURATION) + '/'
	path_train_data = txtdir + 'lms_train_data.txt'
	if not os.path.exists(path_train_data):
		if not os.path.exists(txtdir):
			os.makedirs(txtdir)
		saveLMSFeatures(txtdir, 'train')
		saveLMSFeatures(txtdir, 'test')
	f_train_data = open(path_train_data, 'r')
	f_train_label = open(txtdir + '/' + 'lms_train_label.txt', 'r')
	f_test_data = open(txtdir + '/' + 'lms_test_data.txt', 'r')
	f_test_label = open(txtdir + '/' + 'lms_test_label.txt', 'r')
	train_data = []
	train_label = []
	test_data = []
	test_label = []
	lines1 = f_train_data.readlines()
	lines2 = f_train_label.readlines()
	lines3 = f_test_data.readlines()
	lines4 = f_test_label.readlines()
	print 'train data: ' + str(len(lines1)), 'test data: ' + str(len(lines3))
	print 'reading data...'
	print 'train_data: ', len(lines1)
	i = 0
	for i, line in enumerate(lines1):
		line = line.split(' ')
		train_data.append(line)
		if i % 200 == 0:
			print i
	print 'train_label: ', len(lines2)
	for i, line in enumerate(lines2):
		train_label.append(int(line))
		if i % 200 == 0:
			print i
	print 'test_data: ', len(lines3)
	for i, line in enumerate(lines3):
		line = line.split(' ')
		test_data.append(line)
		if i % 200 == 0:
			print i
	print 'test_label: ', len(lines4)
	for i, line in enumerate(lines4):
		test_label.append(int(line))
		if i % 200 == 0:
			print i
	train_data = np.array(train_data)
	train_label = np.array(train_label)
	test_data = np.array(test_data)
	test_label = np.array(test_label)
	print train_data.shape, train_label.shape, test_data.shape, test_label.shape
	count_train = train_data.shape[0]
	count_test = test_data.shape[0]
	train_data = np.reshape(train_data, [count_train, 128, -1])
	test_data = np.reshape(test_data, [count_test, 128, -1])
	train_data.astype(np.float)
	test_data.astype(np.float)

	train_label1 = np.zeros((count_train, n_classes))
	train_label1[np.arange(count_train), train_label] = 1
	test_label1 = np.zeros((count_test, n_classes))
	test_label1[np.arange(count_test), test_label] = 1
	return train_data, train_label1, test_data, test_label1

def getMFCC(filepath):
	y, sr = librosa.load(filepath)
	mfcc = librosa.feature.mfcc(y=y, sr=sr)
	# mfcc = np.delete(mfcc, len(mfcc[0]) - 1, axis=1)
	# mfcc = np.delete(mfcc, 0, axis=1)
	return mfcc

def saveMFCCFeatures(txtdir, datatype):
	DATA_PATH = ROOT_DIR + str(TIME_DURATION) + '/' + datatype + '/'
	filelist = os.listdir(DATA_PATH)
	f_data = open(txtdir + 'mfcc_' + datatype + '_data.txt', 'w+a')
	f_label = open(txtdir + 'mfcc_' + datatype + '_label.txt', 'w+a')
	i = 0
	for i, filename in enumerate(filelist):
		filepath = DATA_PATH + filename
		mfcc = np.array(getMFCC(filepath))
		c = filename.split('_')[0]
		mfcc = np.reshape(mfcc, [-1])
		mfcc = mfcc.astype(np.str)
		mfcc = ' '.join(mfcc) + '\n'
		f_data.write(mfcc)
		f_label.write(c + '\n')
		if i % 100 == 0:
			print i, ': getMFCC: ' + filepath
	f_data.close()
	f_label.close()

def getMFCCFeatures():
	txtdir = ROOT_DIR + 'data_txt/' + str(TIME_DURATION) + '/'
	path_train_data = txtdir + 'mfcc_train_data.txt'
	if not os.path.exists(path_train_data):
		if not os.path.exists(txtdir):
			os.makedirs(txtdir)
		saveMFCCFeatures(txtdir, 'train')
		saveMFCCFeatures(txtdir, 'test')
		saveMFCCFeatures(txtdir, 'validation')
	f_train_data = open(path_train_data, 'r')
	f_train_label = open(txtdir + '/' + 'mfcc_train_label.txt', 'r')
	f_validation_data = open(txtdir + '/' + 'mfcc_validation_data.txt', 'r')
	f_validation_label = open(txtdir + '/' + 'mfcc_validation_label.txt', 'r')
	f_test_data = open(txtdir + '/' + 'mfcc_test_data.txt', 'r')
	f_test_label = open(txtdir + '/' + 'mfcc_test_label.txt', 'r')
	train_data = []
	train_label = []
	validation_data = []
	validation_label = []
	test_data = []
	test_label = []
	lines1 = f_train_data.readlines()
	lines2 = f_train_label.readlines()
	lines3 = f_test_data.readlines()
	lines4 = f_test_label.readlines()
	lines5 = f_validation_data.readlines()
	lines6 = f_validation_label.readlines()
	print 'train data: ' + str(len(lines1)), 'test data: ' + str(len(lines3)), 'validation data: ' + str(len(lines5))
	print 'reading data...'
	print 'train_data: ', len(lines1)
	i = 0
	for i, line in enumerate(lines1):
		line = line.split(' ')
		train_data.append(line)
		if i % 200 == 0:
			print i
	print 'train_label: ', len(lines2)
	for i, line in enumerate(lines2):
		train_label.append(int(line))
		if i % 200 == 0:
			print i
	print 'test_data: ', len(lines3)
	for i, line in enumerate(lines3):
		line = line.split(' ')
		test_data.append(line)
		if i % 200 == 0:
			print i
	print 'test_label: ', len(lines4)
	for i, line in enumerate(lines4):
		test_label.append(int(line))
		if i % 200 == 0:
			print i
	print 'validation_data: ', len(lines5)
	for i, line in enumerate(lines5):
		line = line.split(' ')
		validation_data.append(line)
		if i % 200 == 0:
			print i
	print 'validation_label: ', len(lines6)
	for i, line in enumerate(lines6):
		validation_label.append(int(line))
		if i % 200 == 0:
			print i
	train_data = np.array(train_data)
	train_label = np.array(train_label)
	test_data = np.array(test_data)
	test_label = np.array(test_label)
	validation_data = np.array(validation_data)
	validation_label = np.array(validation_label)
	print train_data.shape, train_label.shape, validation_data.shape, validation_label.shape, test_data.shape, test_label.shape
	count_train = train_data.shape[0]
	count_test = test_data.shape[0]
	count_validation = validation_data.shape[0]
	train_data = np.reshape(train_data, [count_train, 20, -1])
	test_data = np.reshape(test_data, [count_test, 20, -1])
	validation_data = np.reshape(validation_data, [count_validation, 20, -1])
	train_data.astype(np.float)
	test_data.astype(np.float)
	validation_data.astype(np.float)

	train_label1 = np.zeros((count_train, n_classes))
	train_label1[np.arange(count_train), train_label] = 1
	test_label1 = np.zeros((count_test, n_classes))
	test_label1[np.arange(count_test), test_label] = 1
	validation_label1 = np.zeros((count_validation, n_classes))
	validation_label1[np.arange(count_validation), validation_label] = 1
	return train_data, train_label1, validation_data, validation_label1, test_data, test_label1

if __name__ == '__main__':
	#getLMSFeatures()
	print getMFCC(ROOT_DIR + '3/train/0_0.mp3').shape
	print getMFCC(ROOT_DIR + '5/train/0_0.mp3').shape
	# getMFCC()
	# f = open('mfcc.txt', 'a+w')
	# for i in xrange(1, 7):
	# 	filepath = 'data/splits/2_90_0' + str(i) + '.wav'
	# 	print filepath
	# 	mfcc = getMFCCFeatures(filepath)
	# 	mfcc = mfcc.astype(np.str)
	# 	f.write(filepath + '\n')
	# 	for j in xrange(0, mfcc.shape[0]):
	# 		line = ' '.join(mfcc[j])
	# 		f.write(line + '\n')
	# 	f.write('\n')
	# f.close()