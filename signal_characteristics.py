import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
#from signal_functions_filter import*

from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from openpyxl import Workbook, load_workbook
import tarfile
import glob
from scipy.optimize import curve_fit

import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities.trace_utilities import butterworth_filter_trace, upsampling_fir
from NuRadioReco.utilities import fft, bandpass_filter
from NuRadioReco.utilities import units

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

def gauss(x,  A, x0, sigma):
    return  A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[max(y), mean, sigma])
    return popt, pcov

def plot_V_distr_in_file(V,pdf_file1,pdf_file2):
    """ plot voltage distribution and save in file
    V = array[4][]
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=(5, 10))
    ax = [ax1, ax2, ax3, ax4]
    for i in range(4):
        h = ax[i].hist(V[i],40,color ='blue',histtype = 'step',label='ch'+str(i)+' data')
        ax[i].set_xlabel('max V [mV]',fontsize = 12)
        ax[i].set_ylabel('Number',fontsize = 12)
        ydata = h[0]
        xdata = np.array([(h[1][i]+h[1][i+1])/2 for i in range(len(h[1])-1)])
        ax[i].errorbar(xdata, ydata, yerr=np.sqrt(ydata), ecolor='blue', elinewidth=1, capsize=2,lw = 0, ls = None)
        popt, pcov =  gauss_fit(xdata, ydata)
        perr = np.sqrt(np.diag(pcov))
        A, mean, sigma = popt
        print('channel', i)
        print('A = ',round(A,3),'+-',round(perr[0],3))
        print('mean = ',round(mean,3),'+-',round(perr[1],3))
        print('sigma = ',round(sigma,3),'+-',round(perr[2],3))
        ax[i].plot(xdata, gauss(xdata, *popt), 'r-',label='gauss')#:  $\mu$=%5.1f, $\sigma$=%5.1f' % (mean,sigma))
        ax[i].legend()
        if(np.max(ydata)>1000):
            ax[i].set_yscale('log')
        ax[i].set_ylim(0.5)
    plt.tight_layout()
    pdf_file1.savefig(fig)
    plt.show()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=(5, 10))
    ax = [ax1, ax2, ax3, ax4]
    for i in range(4):
        h = ax[i].hist(V[i],40,color ='blue',histtype = 'step',label='ch'+str(i)+' data')
        ax[i].set_xlabel('max V [mV]',fontsize = 12)
        ax[i].set_ylabel('Number',fontsize = 12)
        ydata = h[0]
        xdata = np.array([(h[1][i]+h[1][i+1])/2 for i in range(len(h[1])-1)])
        popt, pcov =  gauss_fit(xdata, ydata)
        perr = np.sqrt(np.diag(pcov))
        A, mean, sigma = popt
        ax[i].plot(xdata, gauss(xdata, *popt), 'r-',label='gauss')#:  $\mu$=%5.1f, $\sigma$=%5.1f' % (mean,sigma))
        ax[i].legend()

    plt.tight_layout()
    pdf_file2.savefig(fig)
    plt.close()
    y_fit = gauss(xdata, *popt)
    return [xdata,y_fit]

def do_wavelet(series):
    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.1
    j1 = 4 / dj  # this says do 12 powers-of-two with dj sub-octaves each
    lag1 = 0.72  # lag-1 autocorrelation for red noise background
    mother = 'MORLET'

    n = len(series)
    dt = 0.5
    s0 = 2 * dt  # this says start at a scale of 2 ns
    series = series - np.mean(series)
    variance = np.std(series, ddof=1) ** 2
    if 0:
        variance = 1.0
        series = series/ np.std(series , ddof=1)

    # Wavelet transform:
    wave, period, scale, coi = wavelet(series , dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

    # Significance levels:
    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
        lag1=lag1, mother=mother)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
    sig95 = power / sig95  # where ratio > 1, power is significant


    # Global wavelet spectrum & significance levels:
    dof = n - scale  # the -scale corrects for padding at edges
    global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1,
        lag1=lag1, dof=dof, mother=mother)

    # Scale-average between  periods of time1--time2
    avg = np.logical_and(scale >= 3.3, scale < 10)#0.3Ghz-0.1GHz(scale >= np.min(period), scale < np.max(period))
    Cdelta = 0.776  # this is for the MORLET wavelet
    scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand scale --> (J+1)x(N) array
    scale_avg = power / scale_avg  # [Eqn(24)]
    scale_avg = dj * dt / Cdelta * sum(scale_avg[avg, :])  # [Eqn(24)]
    scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2,
        lag1=lag1, dof=([2, 7.9]), mother=mother)

    return power,period,sig95,global_ws,global_signif,scale_avg,scaleavg_signif

def filter_wavelet(series):
    power,period,sig95,global_ws,global_signif,scale_avg,scaleavg_signif = do_wavelet(series)
    filter_wf = np.array([])

    for num,v in enumerate(series):
        if (scale_avg[num]>scaleavg_signif):
            continue
        filter_wf = np.append(filter_wf,v)
    return filter_wf

def notch_maxF(freq,complex_ftt,df):
    spectrum = np.abs(complex_ftt)
    maxF = freq[np.argmax(spectrum)]
    filter_band= channelBandPassFilter.get_filter(freq*units.GHz,station_id = 0, channel_id = 0, det = None, passband = [(maxF-df)* units.GHz, (maxF+df)* units.GHz], filter_type = 'rectangular')
    complex_notch_ftt = complex_ftt-complex_ftt*filter_band
    return complex_notch_ftt

def filter_fft(series,freq):
    complex_ftt = np.fft.rfft(series, norm='ortho')
    for i in range(3):
        complex_ftt = notch_maxF(freq,complex_ftt,0.008)
    filter_notch_wf = np.fft.irfft(complex_ftt,norm='ortho')
    return filter_notch_wf

def sum_plots(data,filter_data, pdf_file):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=(5, 10))
    ax = [ax1, ax2, ax3, ax4]
    for i in range(4):
        h = ax[i].hist(filter_data[i],40,color ='g',histtype = 'step',label='filter data')
        ydata = h[0]
        xdata = np.array([(h[1][i]+h[1][i+1])/2 for i in range(len(h[1])-1)])
        ax[i].errorbar(xdata, ydata, yerr=np.sqrt(ydata), ecolor='g', elinewidth=1, capsize=2,lw = 0, ls = None)
        popt, pcov =  gauss_fit(xdata, ydata)
        perr = np.sqrt(np.diag(pcov))
        A, mean, sigma = popt

        ax[i].plot(xdata, gauss(xdata, *popt), 'r-',label='gauss')

        h2 = ax[i].hist(data[i],40,color ='blue',histtype = 'step',label='ch'+str(i)+' data')
        ydata2 = h2[0]
        ax[i].set_xlabel('max V [mV]',fontsize = 12)
        ax[i].set_ylabel('Number',fontsize = 12)


        ax[i].legend()
        ax[i].set_yscale('log')
        ax[i].set_ylim(0.5,1.5*np.max(ydata2))
    plt.tight_layout()
    pdf_file.savefig(fig)
    pdf_file.close()
    plt.show()

def do_fft(series):
    series-=np.mean(series)
    return np.abs(np.fft.rfft(series, norm='ortho'))
def specrums_plots(freq,spectrum, pdf_file):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=(5, 10))
    ax = [ax1, ax2, ax3, ax4]
    for i in range(4):
        ax[i].plot(freq[i],spectrum[i], lw = 2, label = 'ch'+str(i))
        ax[i].set_xlabel('Frequency [GHz]',fontsize = 12)
        ax[i].set_ylabel('Amplitude [mV]',fontsize = 12)
        ax[i].grid(which='major', color = 'grey')
        ax[i].minorticks_on()
        ax[i].grid(which='minor', color = 'gray', linestyle = ':')
        ax[i].legend()

    plt.tight_layout()
    pdf_file.savefig(fig)
    pdf_file.close()
    plt.show()
def rolling_mean(freq,spectrum,w = 40):
    freq_roll = [np.array([]) for i in range(4)]
    spectrum_roll = [np.array([]) for i in range(4)]
    for ch in range(4):
        fl1 = pd.Series(spectrum[ch]).rolling(window=w).mean()
        #fl2 = np.array([specttrum[ch][i] for i in range(round(w/2), len(specttrum[ch])-round(w/2)+1)])
        spectrum_roll[ch] = np.array([fl1[i] for i in range(w-1, len(fl1))])
        freq_roll[ch] = np.array([freq[ch][i] for i in range(round(w/2), len(freq[ch])-round(w/2)+1)])
    return freq_roll,spectrum_roll
def specrums_plots_two(freq1,spectrum1,freq2,spectrum2, pdf_file, lim = [0,1],size = (5, 10)):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=size)
    ax = [ax1, ax2, ax3, ax4]
    for i in range(4):
        ax[i].plot(freq1[i],spectrum1[i], lw = 2, label = 'ch'+str(i)+' before filter')
        ax[i].plot(freq2[i],spectrum2[i], lw = 2, label = 'ch'+str(i)+' after filter')
        ax[i].set_xlabel('Frequency [GHz]',fontsize = 12)
        ax[i].set_ylabel('Amplitude [mV]',fontsize = 12)
        ax[i].grid(which='major', color = 'grey')
        ax[i].minorticks_on()
        ax[i].grid(which='minor', color = 'gray', linestyle = ':')
        ax[i].legend()
        ax[i].set_xlim(lim)

    plt.tight_layout()
    pdf_file.savefig(fig)
    pdf_file.close()
    plt.show()
def notch_maxF_freq(freq,complex_ftt,df):
    spectrum = np.abs(complex_ftt)
    arg_max = np.argmax(spectrum)
    maxF = freq[arg_max]
    filter_band= channelBandPassFilter.get_filter(freq*units.GHz,station_id = 0, channel_id = 0, det = None, passband = [(maxF-df)* units.GHz, (maxF+df)* units.GHz], filter_type = 'rectangular')
    complex_notch_ftt = complex_ftt-complex_ftt*filter_band
    return complex_notch_ftt, maxF, spectrum[arg_max]
def filter_fft_freq(series,freq):
    series = series-np.mean(series)
    complex_ftt = np.fft.rfft(series, norm='ortho')
    filter_freq = np.array([])
    filter_voltages = np.array([])
    for i in range(3):
        complex_ftt,maxF, maxA = notch_maxF_freq(freq,complex_ftt,0.008)
        filter_freq = np.append(filter_freq,maxF)
        filter_voltages = np.append(filter_voltages,maxA)
    filter_notch_wf = np.fft.irfft(complex_ftt,norm='ortho')
    return filter_notch_wf, filter_freq, filter_voltages
def snr(series):
    return np.max(np.abs(series))/np.std(series)
def snr_ratio(snr_before,snr_after,pdf_file):
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, hspace=0.0, wspace=0.1)
    plt.subplots_adjust(left=0.05, bottom=0.005, right=0.98, top=0.95, wspace=0, hspace=0)

    for ch in range(0,4):

        plt3 = plt.subplot(gs[0, ch])

        h = plt.hist2d(snr_before[ch],snr_after[ch] ,bins = [20,20],cmap='rainbow', norm=LogNorm())
        line = np.array([min(snr_before[ch]),max(snr_before[ch])])
        plt.plot(line,line)
        ax = plt.gca()
        plt.colorbar(h[3], ax=ax,orientation = 'horizontal')
        if (ch==0):
            plt.ylabel('SNR after filter',fontsize = 13)
        plt.xlabel('SNR before filter',fontsize = 13)

        plt.title("ch "+str(ch),fontsize = 13)

        plt.grid(which='major', color = 'grey')
        plt.minorticks_on()
        plt.grid(which='minor', color = 'gray', linestyle = ':')


    #plt.legend()
    pdf_file.savefig(fig)
    pdf_file.close()
    plt.show()

def filter1(idrm):
    freq = [np.array([]) for i in range(4)]
    df = [np.array([]) for i in range(4)]
    if(idrm==1):
        freq[0] = np.array([0.185,0.25,0.268])
        df[0] = np.array([0.020,0.008,0.008])
        freq[1] = np.array([0.175,0.25,0.268])
        df[1] = np.array([0.015,0.008,0.008])
        freq[2] = np.array([0.175,0.255, 0.27])
        df[2] = np.array([0.015,0.008,0.008])
        freq[3] = np.array([0.175,0.255, 0.27])
        df[3] = np.array([0.015,0.008,0.008])
    if(idrm==3):
        freq[0] = np.array([0.2])
        df[0] = np.array([0.010])
        freq[1] = np.array([0.21])
        df[1] = np.array([0.010])
        freq[2] = np.array([0.23])
        df[2] = np.array([0.005])
        freq[3] = np.array([0.13])
        df[3] = np.array([0.010])
    if(idrm==2):
        freq[3] = np.array([0.75,0.955])
        df[3] = np.array([0.020,0.050])
    return freq,df

def filt_fft(series,freq0,freq,df):
    filter_notch_wf = [np.array([]) for i in range(4)]
    for ch in range(4):
        series[ch+1] = series[ch+1]-np.mean(series[ch+1])
        complex_ftt = np.fft.rfft(series[ch+1], norm='ortho')
        for num, f in enumerate(freq[ch]):
            filter_band= channelBandPassFilter.get_filter(freq0*units.GHz,station_id = 0, channel_id = 0, det = None, passband = [(f-df[ch][num])* units.GHz, (f+df[ch][num])* units.GHz], filter_type = 'rectangular')
            complex_ftt = complex_ftt-complex_ftt*filter_band
        filter_notch_wf[ch] = np.fft.irfft(complex_ftt,norm='ortho')
    return filter_notch_wf

def plot_waves_in_file_no_lags(time,traces,pdf_file,title = None, signal_time = [-1,-1,-1,-1]):
    """This function plot waves

    series --- np.array[4][] mV
    time np.array[4] ns
    title --- str
    """
    fig, (ax1,ax2,ax3,ax4)  = plt.subplots(4, 1, sharex=True, figsize=(6, 7))
    ax = (ax1,ax2,ax3,ax4)
    plt.subplots_adjust(hspace = 0,wspace = 0)
    ax[0].set_title(title,fontsize = 13)


    for ch in range(0,4):
        ax[ch].plot(time/units.ns, traces[ch]/units.mV,color='b',lw = 1.2)
        if(signal_time[ch]>=0):
            ax[ch].axvline(signal_time[ch])
        ax[ch].grid(which='major', color = 'grey')
        ax[ch].minorticks_on()
        ax[ch].grid(which='minor', color = 'gray', linestyle = ':')


    ax[3].set_xlabel('Time, ns',fontsize = 13)
    ax[0].set_ylabel('Voltage, mV',fontsize = 13)
    plt.tight_layout()
    pdf_file.savefig(fig)
    plt.close()
def specrums_plots_in_file(freq,spectrum, pdf_file,title = None):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,figsize=(5, 10))
    ax = [ax1, ax2, ax3, ax4]
    ax[0].set_title(title,fontsize = 13)
    for i in range(4):
        ax[i].plot(freq,spectrum[i], lw = 2, label = 'ch'+str(i))
        ax[i].set_xlabel('Frequency [GHz]',fontsize = 12)
        ax[i].set_ylabel('Amplitude [mV]',fontsize = 12)
        ax[i].grid(which='major', color = 'grey')
        ax[i].minorticks_on()
        ax[i].grid(which='minor', color = 'gray', linestyle = ':')
        ax[i].legend()

    plt.tight_layout()
    pdf_file.savefig(fig)
    plt.close()

def S_spectrum1(spectrum):
    S = 0
    for a in spectrum:
            S+=a*a
    return S
def fraction_power_exclude(wf,wf_filter ):
    filter_spectrum = np.abs(np.fft.rfft(wf_filter, norm='ortho'))
    spectrum = np.abs(np.fft.rfft(wf, norm='ortho'))
    s0 = S_spectrum1(spectrum)
    s1 = S_spectrum1(filter_spectrum)
    return (s0-s1)/s0
def fraction_power_exclude_by_spectrums(spectrum,filter_spectrum ):
    s0 = S_spectrum1(spectrum)
    s1 = S_spectrum1(filter_spectrum)
    return (s0-s1)/s0

def filter_fft_n(series,freq,n_iter = 3):
    complex_ftt = np.fft.rfft(series, norm='ortho')
    for i in range(n_iter):
        complex_ftt = notch_maxF(freq,complex_ftt,0.008)
    filter_notch_wf = np.fft.irfft(complex_ftt,norm='ortho')
    return filter_notch_wf
