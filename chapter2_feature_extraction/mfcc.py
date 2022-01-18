import librosa
import numpy as np
from scipy.fft import dct
import sys

# If you want to see the spectrogram picture
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_spectrogram(spec, note, file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """ 
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    #plt.show()
    plt.savefig(file_name)

def plot_raw_wave(signal, num=0):
    x = np.linspace(0, signal.size, signal.size)
    plt.figure(num, figsize=(20, 5))
    plt.plot(x, signal)
    plt.xlabel("samples")
    plt.ylabel("amplitude")

#preemphasis config 
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=16kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12
#print(mel_high, type(mel_high))

# Generate hz frequnces (begin center stop) 
mel_low = 0
mel_high = 2595*np.log10(1+(8000/700))
mel_freq_band = (mel_high-mel_low)/(num_filter+1)
mel_center_freq = mel_low + np.array(range(0, num_filter+2)) * mel_freq_band # total 25 mel, index[0,24]
hz_center_freq = (np.power(10, mel_center_freq/2595) - 1) * 700 # mel to hz
hz_center_index = np.floor(hz_center_freq / (16000/512))
#print(hz_center_freq)
#print(hz_center_index)
# print(np.floor(hz_center_index))

# Read wav file
#wav, fs = librosa.load('./test.wav', sr=None)

# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] * win
        # frames[i,:] = frames[i,:] 

    return frames

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    # print(cFFT.shape)
    # print(cFFT[0], cFFT[0].shape)
    # print("index000", np.abs(cFFT[0][0]))
    # print("index001", np.abs(cFFT[0][1]))
    # print("index002", np.abs(cFFT[0][2]))
    # print("index255", np.abs(cFFT[0][255]))
    # print("index256", np.abs(cFFT[0][256]))
    # print("index257", np.abs(cFFT[0][257]))
    # print("index510", np.abs(cFFT[0][510]))
    # print("index511", np.abs(cFFT[0][511]))
    # plt.plot(range(512), np.abs(cFFT[0]))
    # plt.show()
    valid_len = int(fft_len/2) + 1 # 257 samples
    spectrum = np.abs(cFFT[:,0:valid_len])
    return spectrum

def compute_mel_filter(num_filter=num_filter, fft_len=fft_len):
    """Generate mel triangle Filter Bank 
        :param num_filter: mel filters number, default 23
        :param fft_len: FFT length, default 512
        :returns: mel_filter, a num_filter by fft_len/2+1 array
    """
    hz_filter = np.zeros((num_filter, int(fft_len/2) + 1))
    for m in range(1,num_filter+1):
        fm1 = int(hz_center_index[m-1])
        fm = int(hz_center_index[m])
        fm2 = int(hz_center_index[m+1])
        for k in range(fm1, fm+1):
            hz_filter[m-1,k] = (k-fm1)/(fm-fm1)
        for k in range(fm+1, fm2+1):
            hz_filter[m-1,k] = (fm2-k)/(fm2-fm)
    #print(hz_filter, hz_filter.shape)
    return hz_filter

def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """
    feats=np.zeros((spectrum.shape[0], num_filter))
    feats = np.dot(spectrum, compute_mel_filter().T)
    return np.log10(feats)

def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """
    #feats = np.zeros((fbank.shape[0],num_mfcc))
    """
        FINISH by YOURSELF
    """
    return dct(fbank, type=2, axis=1, norm='ortho')[:,1:num_mfcc+1]

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('./test.wav', sr=None) # 356 frames this wav file
    #print(wav, type(wav), wav.shape)
    signal = preemphasis(wav)
    #print(signal, signal.shape)
    # plot_raw_wave(wav,1)
    # plot_raw_wave(signal,2)
    # plt.show()
    
    frames = enframe(signal)
    #print(frames, frames.shape)
    spectrum = get_spectrum(frames)
    #print(spectrum, spectrum.shape)
    
    fbank_feats = fbank(spectrum)
    print(fbank_feats, fbank_feats.shape)
    mfcc_feats = mfcc(fbank_feats)
    print(mfcc_feats, mfcc_feats.shape)
    plot_spectrogram(fbank_feats.T, 'Filter Bank','fbank.png')
    write_file(fbank_feats,'./test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()
