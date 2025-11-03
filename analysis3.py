from sklearn.metrics import accuracy_score
import numpy as np
import scipy.io.wavfile as wav #for sound file
import os, glob
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

from python_speech_features import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
import dsp

data_folder = '../data/Fully_Segmented/'

name = []
y = []
X = []
a = []

plt.close("all")

#np.random.seed(200)

for filename in glob.glob(os.path.join(data_folder, '*.wav')):
	(head, tail) = os.path.split(filename)
	name.append(tail)


y = []

for i in range(len(name)):
	num = i

	file= os.path.join(data_folder,name[i])


	(fs, data) = wav.read(file)

	# use left channel
	x = data[:, 0]

	winlen= 20e-3
	winstep=  10e-3
	numcep=13
	nfilt=26
	nfft=1024
	lowfreq=0
	highfreq= (fs/2)
	preemph=0.97
	ceplifter=22
	appendEnergy=True
	winfunc=np.hamming

	###A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
	#feat= mfcc(x,fs,winlen,winstep,numcep,nfilt,nfft,lowfreq,highfreq,preemph,ceplifter,appendEnergy,winfunc)
	#feat = fbank(x,fs,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)[0]
	feat = ssc(x,fs,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)

	dfeat = delta(feat, 2)



	# compute mean and standard deviation
	#feat = np.hstack((feat,dfeat))
	fmean = np.mean(feat, axis = 0)
	fstd = np.std(feat, axis = 0)

	#print(mfcc_feat.shape)

	if file[-5] == 'p':
		y.append(1)  # aspirating case is 1
	else:
		y.append(0)

	if i == 0:
		featureMatrix = np.hstack((fmean, fstd))
	else:
		featureMatrix = np.vstack((featureMatrix, np.hstack((fmean, fstd))))


x = np.copy(featureMatrix)
(N,fn) = x.shape
Na = np.sum(y)
Nn = N - Na
fn >>=1
nasp = np.zeros((Nn, fn))
asp = np.zeros((Na, fn))

j = 0;
k = 0;
for i in range(N):
	if y[i] == 0:
		nasp[j, :] = x[i, 0:fn]
		j += 1
	else:
		asp[k, :] = x[i, 0:fn]
		k += 1
	
# plot the SSCs
plt.figure()
plt.subplot(211)
ind = np.random.permutation(106)
nasp = nasp[ind,:]
for i in range(106):
	for j in range(fn):
		plt.tick_params(labelleft=False,left=False)
		xp = [nasp[i, j], nasp[i, j]]
		plt.plot(xp, [0, 1], "r-")
plt.xlabel("Frequency (Hz)")
plt.title("Mean spectral subband centroids for all non-aspirating patients")

plt.grid()
ind = np.random.permutation(18)
asp = asp[ind,:]
plt.subplot(212)		
for i in range(18):
	for j in range(fn):
		plt.tick_params(labelleft=False,left=False)
		xp = [asp[i, j], asp[i, j]]
		plt.plot(xp, [0, 1], "b-")
plt.xlabel("Frequency (Hz)")
plt.tick_params(left=False)
plt.title("Mean spectral subband centroids for all aspirating patients")
plt.tight_layout()
plt.grid()
plt.show(block=False)

"""
for i in range(18):
	plt.figure()
	for j in range(fn):
		xp = [asp[i, j], asp[i, j]]
		plt.plot(xp, [0, 1], "b-")
		plt.tight_layout()
		plt.grid()
		plt.show(block=False)
"""