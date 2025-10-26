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

#SVC Classifier
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.5, random_state = 17, stratify = y) # 5, 9, 17

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf','poly','sigmoid'], 'degree':[2,3,4,5], 'gamma': [0.5, 1e-1, 1e-2, 2e-2, 5e-2, 1e-3, 1e-4],
                     'C': [1, 10, 20, 50, 100, 1000], 'class_weight': [None,'balanced']}]
                    #{'kernel': ['linear'], 'gamma': [0.01], 'C': [1, 10, 50, 100, 1000], 'class_weight': [None,'balanced']}]

scores = ['precision']

for score in scores:
	print("# Tuning hyper-parameters for %s" % score)
	print()

	clf = GridSearchCV(
		SVC(), tuned_parameters, n_jobs=4, cv = 5  # perform a stratified cross-validation
	)
	clf.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print()
	#print("Grid scores on development set:")
	#print()
	#means = clf.cv_results_['mean_test_score']
	#stds = clf.cv_results_['std_test_score']
	#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	#	print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
	#print()

	print("Detailed classification report:")
	print()
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print()
	#y_true, y_pred = y_test, clf.predict(X_test)
	svm = SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], kernel=clf.best_params_['kernel'], class_weight="balanced")
	svm.fit(X_train, y_train)
	(y_true, y_pred) = (y_test, svm.predict(X_test))
	print(classification_report(y_true, y_pred))
	print()

	print("Confusion matrix")
	print(confusion_matrix(y_true, y_pred))
	plot_confusion_matrix(svm, X_test, y_test, display_labels=["Normal", "Aspirating"])
	plt.show(block=False)
	print("Precision (rate of correct predictions for + predictions) :", precision_score(y_true, y_pred))
	print("Recall/sensitivity (rate of correct predictions for + samples):", recall_score(y_true, y_pred))
	print("Accuracy:", accuracy_score(y_true, y_pred))
	print("F1 score:", f1_score(y_true, y_pred))

	print("y_true", np.array(y_true))
	print("y_pred", y_pred)
