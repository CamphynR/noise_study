import numpy as np

def db_coeff(db):
    return 10.**(db/20.)

class dbAmplifier():
    """
    Basic module to amplify signal by db for testing purposes
    """
    def __init__(self, db=55, reduce=False):
        self.db = db
        self.db_coeff = db_coeff(db)
        self.reduce = reduce
        return

    def run(self, event, station, detector):
        for channel in station.iter_channels():
            trace = channel.get_trace()
            if not self.reduce:
                trace = trace * self.db_coeff
            else:
                trace = trace / self.db_coeff
            channel.set_trace(trace, channel.get_sampling_rate())
