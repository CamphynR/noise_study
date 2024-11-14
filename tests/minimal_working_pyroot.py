import mattak.Dataset

ds = mattak.Dataset.Dataset(station = 0, run = 0, data_path  ="~/Documents/data/RNO-G_data/station23/run1603",
                            verbose = True, backend = "pyroot")


for idx, (evtinfo, wf) in enumerate(ds.iterate(calibrated = True)):
    print(wf) # or do something else