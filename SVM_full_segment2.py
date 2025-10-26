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
import h5py as h5
import kmeans

plt.close("all")

classes = ["aspirating", "normal"]
fileHandle = h5.File("fs_db.h5", "r")
fs = 44100

# feature extraction parameters
winlen= 20e-3
winstep= 10e-3
numcep=13
nfilt=26
nfft=2048
lowfreq=0
highfreq= (fs/2)
preemph=0.97
ceplifter=22
appendEnergy=True
winfunc=np.hamming

y = []
i = 0

for c in classes:
	for wave in fileHandle[c]:
		data = fileHandle[c][wave][...]
		
		# use left channel
		x = data[:, 0]

		#feat= mfcc(x,fs,winlen,winstep,numcep,nfilt,nfft,lowfreq,highfreq,preemph,ceplifter,appendEnergy,winfunc)
		#feat = fbank(x,fs,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)[0]
		feat = ssc(x,fs,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)


		#dfeat = delta(feat, 2)
		# compute mean and standard deviation
		#feat = np.hstack((feat,dfeat))
		
		fmean = np.mean(feat, axis = 0)
		fstd = np.std(feat, axis = 0)

		#km = kmeans.Kmeans(nfilt, 2)
		#km.train(np.transpose(feat), maxIter = 30)
		
		
		#fmean = np.hstack(km.means)

		if c == "aspirating":
			y.append(1)  # aspirating case is 1
		else:
			y.append(0)

		if i == 0:
			featureMatrix = np.hstack((fmean, fstd))
			#featureMatrix = fmean
		else:
			featureMatrix = np.vstack((featureMatrix, np.hstack((fmean, fstd))))
			#featureMatrix = np.vstack((featureMatrix, fmean))
		
		i += 1
		
fileHandle.close()

x = np.copy(featureMatrix)

np.random.seed(1)

# perform multiple times
trials = 20
prec = np.zeros(trials)
rec = np.zeros(trials)
acc = np.zeros(trials)
f1 = np.zeros(trials)

conAvg = np.zeros((2, 2))

for r in range(trials):

	print("Trial {}:".format(r))

	#SVC Classifier
	(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, stratify = y) # 5, 9, 17

	# Set the parameters by cross-validation
	tuned_parameters = [{'kernel': ['rbf','poly','sigmoid'], 'degree':[2, 3,4], 'gamma': [0.5, 0.25, 1e-1, 1e-2, 2e-2, 5e-2, 1e-3, 1e-4],
		                 'C': [1, 10, 20, 50, 100, 1000], 'class_weight': [None,'balanced']},
		                {'kernel': ['linear'], 'gamma': [0.5, 0.25, 0.01], 'C': [1, 10, 50, 100, 1000], 'class_weight': [None,'balanced'], 'degree':[1]}]

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
		print("Best scores from CV:")
		print(np.max(clf.cv_results_['mean_test_score']))
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
		svm = SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], kernel=clf.best_params_['kernel'], class_weight="balanced", degree=clf.best_params_['degree'])
		svm.fit(X_train, y_train)
		(y_true, y_pred) = (y_test, svm.predict(X_test))
		print(classification_report(y_true, y_pred))
		print()

		print("Confusion matrix")
		conf = confusion_matrix(y_true, y_pred) 
		print(conf)
		
		conAvg += conf
		#plot_confusion_matrix(svm, X_test, y_test, display_labels=["Normal", "Aspirating"])
		plt.show(block=False)
		print("Precision (rate of correct predictions for + predictions) :", precision_score(y_true, y_pred))
		print("Recall/sensitivity (rate of correct predictions for + samples):", recall_score(y_true, y_pred))
		print("Accuracy:", accuracy_score(y_true, y_pred))
		print("F1 score:", f1_score(y_true, y_pred))

		print("y_true", np.array(y_true))
		print("y_pred", y_pred)
		
		prec[r] = precision_score(y_true, y_pred)
		rec[r] = recall_score(y_true, y_pred)
		acc[r] = accuracy_score(y_true, y_pred)
		f1[r] = f1_score(y_true, y_pred)

print("Average results over {} trials:".format(trials))
print("Precision: {:.5f} std {:.5f}".format(np.mean(prec), np.std(prec)))
print("Recall: {:.5f} std {:.5f}".format(np.mean(rec), np.std(rec)))
print("Accuracy: {:.5f} std {:.5f}".format(np.mean(acc), np.std(acc)))
print("F1: {:.5f} std {:.5f}".format(np.mean(f1), np.std(f1)))

print("Average confusion matrix:")
print(conAvg)
