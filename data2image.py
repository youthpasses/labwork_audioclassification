# coding:utf-8
# @makai
# 16/10/29

import os
import librosa
import Image
import numpy as np

TRAIN_DATA_DIR = 'data/data_train/'
TRAIN_IMAGE_DIR = 'data/image_train/'
TEST_DATA_DIR = 'data/data_test/'
TEST_IMAGE_DIR = 'data/image_test/'

if not os.path.exists(TRAIN_IMAGE_DIR):
	os.mkdir(TRAIN_IMAGE_DIR)
if not os.path.exists(TEST_IMAGE_DIR):
	os.mkdir(TEST_IMAGE_DIR)

def data2image(oriDir, desDir):
	filelist = os.listdir(oriDir)
	filelist.sort()
	for i, filename in enumerate(filelist):
		filepath = oriDir + filename
		imagename = filename.split('.')[0] + '.png'
		imagepath = desDir + imagename
		y, sr = librosa.load(filepath)
		ms = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, fmax=8000)
		ms = -librosa.logamplitude(ms, ref_power=np.max)
		ms = 255 * (1 - ms / float(np.max(ms)))
		ms = np.delete(ms, len(ms[0]) - 1, axis=1)
		ms = np.delete(ms, 0, axis=1)
		ms = ms[::2,::2]
		# ms = np.reshape(ms, (64 * 64))
		# ms = [0 if x <= 100 else x for x in ms]
		# ms = np.reshape(ms, (64, 64))
		image = Image.fromarray(ms)
		if image.mode != 'L':
			image = image.convert('L')
		image.save(imagepath)
		if i % 100 == 0:
			print i, imagepath

data2image(TRAIN_DATA_DIR, TRAIN_IMAGE_DIR)
data2image(TEST_DATA_DIR, TEST_IMAGE_DIR)
