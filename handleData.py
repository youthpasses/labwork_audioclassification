# coding:utf-8
# @makai
# 16/11/15

import os
import shutil
import random
import librosa

ORIGIN_DIR = 'data/add5classes/'
DES_DIR = 'data/add5classes_piece/'

def normalizeMp3(filedir):
	filelist = os.listdir(filedir)
	filelist.sort()
	for filename in filelist:
		filepath = filedir + filename
		y, sr = librosa.load(filepath)
		y = librosa.to_mono(y)
		y = librosa.util.normalize(y)
		librosa.output.write_wav(filepath, y, 22050)
		print 'Normalize: ' + filepath

def handleMp3(filedir, index):
	filelist = os.listdir(filedir)
	filelist.sort()
	i = 0
	for filename in filelist:
		filepath = filedir + filename
		y, sr = librosa.load(filepath)
		if len(y) < 22050:
			return
		for j in xrange(0, (len(y) - 22050 * 2) / 22050):
			yy = y[j * 22050 : (j + 3) * 22050]
			librosa.output.write_wav(DES_DIR + str(index) + '_' + str(i) + '.mp3', yy, 22050)
			i += 1
		print 'HandleMp3: ' + filepath

def handleData():
	dirlist = os.listdir(ORIGIN_DIR)
	dirlist.sort()
	for dirpath in dirlist:
		path = ORIGIN_DIR + dirpath + '/'
		normalizeMp3(path)
		handleMp3(path,dirpath)
		print '\n'

if __name__ == '__main__':
	handleData()