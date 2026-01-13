import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

from NuRadioReco.utilities.fft import time2freq, freq2time


class flowerDataset():
    def __init__(self, runpath):
        """
        filepath : path to run file containing flower data
        """
        # hardcoded flower specifics
        self.nr_bits = 8
        self.voltage_range = 2
        self.nr_samples = 1024
        self.nr_channels = 4
        self.sampling_rate = 0.472


        flower_path = runpath + "/aux" + "/flower_end.json"
        with open(flower_path, "r") as file:
            content_dic = json.load(file)

        print(content_dic.keys())
        print(content_dic["events"][0].keys())
        print(content_dic["events"][0]["metadata"].keys())


        self.run = content_dic["run"]
        self.station_id = content_dic["hostname"].split("rno-g-0")[1]
        self.nr_events = len(content_dic["events"])

        runinfo_path = runpath + "/aux" + "/runinfo.txt"

        with open(runinfo_path, "r") as file:
            for line in file:
                if line.startswith("RUN-START-TIME"):
                    self.run_start_time = float(line.split("=")[1])
                if line.startswith("RUN-END-TIME"):
                    self.run_end_time = float(line.split("=")[1])

        flower_gain_path = runpath + "/aux" + "/flower_gain_codes.0.txt"

        self.wfs = []
        for event in content_dic["events"]:
            wf_ch = []
            for channel_id in range(self.nr_channels):
                wf_ch.append(event["ch"+str(channel_id)])
            self.wfs.append(wf_ch)
        self.wfs = np.array(self.wfs)

        self.apply_voltage_calibration()

    def apply_voltage_calibration(self, linear=True):
        volts_per_adc = self.voltage_range/(2**self.nr_bits - 1)
        if linear:
            self.wfs = self.wfs * adc_to_v
        else:
            
            selfs.wfs = (self.wfs * adc_to_v) / digital_amplification


    def save_average_spectrum(self, filename, filt=None, debug=False):
        frequencies = np.fft.rfftfreq(self.nr_samples, d=1./self.sampling_rate)
        
        spectra = time2freq(self.wfs, self.sampling_rate)
        if filt is not None:
            spectra = spectra * np.abs(filt)
        spectra[:,:,0] = 0

        if debug:
            channel_id = 1
            plt.plot(frequencies, np.abs(spectra)[0][channel_id], label = "event 0")
            plt.plot(frequencies, np.mean(np.abs(spectra), axis=0)[channel_id], label="run mean")
            plt.legend()
            plt.xlabel("freq / GHz")
            plt.ylabel("spectral amplitude / V/GHz")
            plt.title(f"run {self.run}, channel {channel_id}")
            plt.savefig("figures/tests/test_flower_spectrum.png")

        average_ft = np.mean(np.abs(spectra), axis=0)
        var_average_ft = np.var(np.abs(spectra), axis=0)

        header_dic = {"nr_events" : self.nr_events,
                      "begin_time" : self.run_start_time,
                      "end_time" : self.run_end_time}
        save_dictionary = {"header" : header_dic,
                           "freq" : frequencies,
                           "frequency_spectrum" : average_ft,
                           "var_frequency_spectrum" : var_average_ft}

        with open(filename, "wb") as file:
            pickle.dump(save_dictionary, file)

    def plot_wf(self, wf_idx, channel_id, ax):
        wf = self.wfs[wf_idx][channel_id]
        ax.plot(wf)
        ax.set_xlabel("sample")
        ax.set_ylabel("V")
        ax.set_title(f"run {self.run}")
        plt.show()
 


class flowerDataReader():
    def __init__(self):
        return
    
    def begin(self, run_paths, run_time_range):
        runs = []
        for run_path in run_paths:
            flower_ds = flowerDataset(run_path)
            if start_time < flower_ds.run_start_time < end_time:
                runs.append(flower_ds)
                
