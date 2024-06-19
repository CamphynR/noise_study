import uproot
import numpy as np

file = "/pnfs/iihe/rno-g/data/handcarry22/station24/run1/daqstatus.root"
wf_file = "/pnfs/iihe/rno-g/data/handcarry22/station24/run1/waveforms.root"
file2 = "/pnfs/iihe/rno-g/data/handcarry22/station24/run10/daqstatus.root"

read_data_example_file = "/user/rcamphyn/envs/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/pulser_data_21.root"

print("File from handcarry22, station24 \n--------------------------------")
with uproot.open(file) as f:
    radiant_thresholds = np.array(f["daqstatus"]["radiant_thresholds[24]"])
    lt_trigger_thresholds = np.array(f["daqstatus"]["lt_trigger_thresholds[4]"])
    print("daq")
    print(radiant_thresholds.shape)
    print(lt_trigger_thresholds.shape)
    print(radiant_thresholds.shape[0] + lt_trigger_thresholds.shape[0])
    # print("daq one data example")
    # print(radiant_thresholds[0][1])
    # print(lt_trigger_thresholds[0][1])

with uproot.open(wf_file) as f:
    print("data")
    print(np.array(f["waveforms"]["radiant_data[24][2048]"]).shape)

print("Second file from handcarry22, station24 \n---------------------------------------")
with uproot.open(file2) as f:
    print("daq")
    print(np.array(f["daqstatus"]["radiant_thresholds[24]"]).shape)

print("Root file from read RNOG data example in NuRadioReco \n----------------------------------------------------")
with uproot.open(read_data_example_file) as f:
    print("daq")
    print(np.array(f["combined/daqstatus/radiant_thresholds[24]"]).shape)

with uproot.open(read_data_example_file) as f:
    print("data")
    print(np.array(f["combined/waveforms/radiant_data[24][2048]"]).shape)
