import datetime
from NuRadioReco.detector import detector

if __name__ == "__main__":
    det = detector.Detector(source = "rno_mongo",
                                 always_query_entire_description = False,
                                 database_connection = "RNOG_public",
                                 select_stations = 24)
    det.update(datetime.datetime(2024, 6, 4))
    response = det.get_signal_chain_response(station_id = 24, channel_id = 0)