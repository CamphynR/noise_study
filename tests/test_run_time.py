import argparse
import datetime
import mattak.Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", type=int)
    parser.add_argument("--run", "-r", type=int)
    args = parser.parse_args()

    ds = mattak.Dataset.Dataset(station=args.station, run=args.run, backend="pyroot",verbose=True)
    ds.setEntries(0)
    time = ds.eventInfo().triggerTime
    time_utc = datetime.datetime.fromtimestamp(time).strftime("%d/%m/%Y")

    triggers = ds.eventInfo().triggerType
    print(time_utc)
    print(triggers)
