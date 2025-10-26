# This version uses K-fold cross-validation on evaluation

from sklearn.metrics import accuracy_score
import numpy as np
import scipy.io.wavfile as wav #for sound file
import os, glob
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

from python_speech_features import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
###ACCESS DATA FILES FROM PERSONAL COMPUTER - For 'Segmented Swallowing sounds'
#data_folder = 'C:\\Users\\Jade\\Desktop\\THESIS\\Wav Files\\Fully_Segmented\\'
data_folder = '../data/Fully_Segmented/'
###ACCESS DATA FILES FROM PERSONAL COMPUTER - For ' Fully Segmented Swallowing
### sounds'
#data_folder = 'C:\\Users\\Jade\\Desktop\\THESIS\\Wav Files\\Fully_Segmented\\'
#CREATE ARRAY OF HEADINGS FOR TITLE USE
name = []
y = []
X = []
a = []

plt.close("all")

#np.random.seed(200)

for filename in glob.glob(os.path.join(data_folder, '*.wav')):
	head, tail = os.path.split(filename)
	name.append(tail)


#print(name)

y = []

for i in range(len(name)):
	num = i

	file= os.path.join(data_folder,name[i])

	#print(file)

	(fs, data) = wav.read(file)

	# use left channel
	x = data[:, 0]

	winlen= 20e-3
	winstep=  10e-3
	numcep=13
	nfilt=26
	nfft=2048
	lowfreq=0
	highfreq= (fs/2)
	preemph=0.97
	ceplifter=22
	appendEnergy=True
	winfunc=np.hamming

	###A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
	#feat= mfcc(x,fs,winlen,winstep,numcep,nfilt,nfft,lowfreq,highfreq,preemph,ceplifter,appendEnergy,winfunc)

	feat = ssc(x,fs,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)

	#dfeat = delta(feat, 1)

	#mfcc_feat = logfbank(x,fs,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)

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

# apply standardization
#scaler = preprocessing.StandardScaler().fit(featureMatrix)

#x = scaler.transform(featureMatrix)
x = np.copy(featureMatrix)

#SVC Classifier
#(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.5, random_state = 5)
svm = SVC(C=1, gamma='auto', kernel='poly', class_weight="balanced")

scoring = ['accuracy','precision','recall','f1']
scores = cross_validate(svm, x, y, cv = 10, scoring=scoring, n_jobs=4, verbose=True)

print("Precision: mean={:.4f}, std={:.4f}".format(np.mean(scores['test_precision']), np.std(scores['test_precision'])))
print("Recall: mean={:.4f}, std={:.4f}".format(np.mean(scores['test_recall']), np.std(scores['test_recall'])))
print("F1: mean={:.4f}, std={:.4f}".format(np.mean(scores['test_f1']), np.std(scores['test_f1'])))
print("Accuracy: mean={:.4f}, std={:.4f}".format(np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
