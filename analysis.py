import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io.wavfile as wav
import python_speech_features as psf
import dsp


# No need for this code

plt.close("all")

dataPath = "../data/Fully_Segmented"

aspFile = dataPath + "/MBS15Asp.wav"
naspFile = dataPath + "/P15Norm_SW1_subs0of1.wav"

(fs, x) = wav.read(aspFile)
(fs, xn) = wav.read(naspFile)
# Sampling period
Ts = 1 / fs

# extract left channel
x0 = x[:, 0]
xn0 = xn[:, 0]
Nx = len(x0)
Nxn = len(xn0)
Nmax = np.max(x0)
Nmin = np.min(x0)

Nmaxn = np.max(xn0)
Nminn = np.min(xn0)


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
feat = psf.ssc(x0,fs,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
featn = psf.ssc(xn0,fs,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)

print("Number of points: {}".format(Nx))
print("Minimum: {}".format(Nmin))
print("Maximum: {}".format(Nmax))

t = np.arange(Nx) * Ts
tn = np.arange(Nxn) * Ts



# choose start times
mark = 0.022
markn = 0.07

mark2 = 0.2
mark2n = 0.2

mark3 = 1.02
mark3n = 0.574

plt.subplot(211)
plt.plot(t, x0), plt.xlabel("Time (s)"), plt.ylabel("Amplitude"), plt.grid(), plt.title("(a)")

plt.plot([mark, mark], [0.9*Nmin, 0.8 * Nmax], 'm--')
plt.plot([mark+20e-3, mark+20e-3], [0.9*Nmin,0.8* Nmax], 'm--')
plt.text(0.1*mark, 0.9*Nmax, "(A)")

plt.plot([mark2, mark2], [0.9*Nmin, 0.8 * Nmax], 'm--')
plt.plot([mark2+20e-3, mark2+20e-3], [0.9*Nmin,0.8* Nmax], 'm--')
plt.text(0.8*mark2, 0.9*Nmax, "(B)")

plt.plot([mark3, mark3], [0.9*Nmin, 0.8 * Nmax], 'm--')
plt.plot([mark3+20e-3, mark3+20e-3], [0.9*Nmin,0.8* Nmax], 'm--')
plt.text(mark3, 0.9*Nmax, "(C)")

plt.ylim([Nmin, 1.1*Nmax])
plt.subplot(212)
plt.plot(tn, xn0), plt.xlabel("Time (s)"), plt.ylabel("Amplitude"), plt.grid(),plt.title("(b)")

plt.plot([markn, markn], [0.9*Nminn, 0.7 * Nmaxn], 'm--')
plt.plot([markn+20e-3, markn+20e-3], [0.9*Nminn,0.7* Nmaxn], 'm--')
plt.text(0.5*markn, 0.9*Nmaxn, "(A)")

plt.plot([mark2n, mark2n], [0.9*Nminn, 0.7 * Nmaxn], 'm--')
plt.plot([mark2n+20e-3, mark2n+20e-3], [0.9*Nminn,0.7* Nmaxn], 'm--')
plt.text(0.9*mark2n, 0.9*Nmaxn, "(B)")

plt.plot([mark3n, mark3n], [0.9*Nminn, 0.7 * Nmaxn], 'm--')
plt.plot([mark3n+20e-3, mark3n+20e-3], [0.9*Nminn,0.7* Nmaxn], 'm--')
plt.text(0.99*mark3n, 0.9*Nmaxn, "(C)")

plt.tight_layout()

plt.savefig("time.svg")

# perform some AMS
Tw = 20e-3  # frame size
Tu = 10e-3  # update



frame = int(mark / Tu)
framen = int(markn / Tu)

frame2 = int(mark2 / Tu)
frame2n = int(mark2n / Tu)

frame3 = int(mark3 / Tu)
frame3n = int(mark3n / Tu)

Nw = int(Tw / Ts)
dw = int(Tu / Ts)

# apply preemphasis filter
y0 = sig.lfilter([1, -0.97], 1, x0)
yn0 = sig.lfilter([1, -0.97], 1, xn0)

(xf, Nf) = dsp.frame_split(y0, Nw, dw)
(xnf, Nnf) = dsp.frame_split(yn0, Nw, dw)


def plotpsd(frame, framen, gtitle, gtitle2):
	Nfft = 2048
	#for frame in range(20):
		#P = dsp.periodogram(xf[frame, :], Nfft)
	P = dsp.welchPeriodogram(xf[frame, :], Nfft, int(Nw/3), int(dw/10), wind = 1)
	Pn = dsp.welchPeriodogram(xnf[framen, :], Nfft, int(Nw/3), int(dw/10), wind = 1)
	f = np.arange(Nfft) / Nfft * fs
	centre = int(Nfft / 2 + 1)
	f = f[0 : centre]
	P = P[0 : centre]
	Pn = Pn[0 : centre]
	maxP = np.max(20*np.log10(P))
	minP = np.min(20*np.log10(P))
	maxPn = np.max(20*np.log10(Pn))
	minPn = np.min(20*np.log10(Pn))

	plt.figure()
	plt.subplot(211)
	plt.plot(f, 20 * np.log10(P))
	for s in range(26):
		loc = feat[frame, s]
		plt.plot([loc, loc],[minP, maxP], 'r--')
	title = "(a)".format(frame * Tu)
	plt.title(gtitle)
	plt.grid()

	plt.xlabel("Frequency (Hz)")
	plt.ylabel("Power (dB)")
	plt.tight_layout()

	plt.subplot(212)

	plt.plot(f, 20 * np.log10(Pn))
	for s in range(26):
		loc = featn[framen, s]
		plt.plot([loc, loc],[minPn, maxPn], 'r--')
	title = "Power spectrum of non-aspirating swallow (P15)".format(frame * Tu)
	plt.title(gtitle2)
	plt.grid()
	plt.xlabel("Frequency (Hz)")
	plt.ylabel("Power (dB)")
	plt.tight_layout()
	plt.show(block=False)


plotpsd(frame,framen, "(a)", "(b)")
plt.savefig("psd_a.svg")
plotpsd(frame2,frame2n,  "(a)", "(b)")
plt.savefig("psd_b.svg")
plotpsd(frame3,frame3n,  "(a)", "(b)")
plt.savefig("psd_c.svg")