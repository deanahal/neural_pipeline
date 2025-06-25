'''
main neural pipeline script:
this script orchestrates the loading of data, filtering, feature extraction, and classification.

offline data is loaded from .ns5 and .nev files (either Ripple or BlackRock)
filters are either applied or taken from already filtered data (only Ripple)
data is inserted into a classifier 


'''
from ripple_io.ripple_loader import RippleLoader
#from processing.filter import custom_filter
#from models.classifier import EventClassifier
import importlib
import sys
import os
import ripple_io
import ripple_io.ripple_loader

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(r"C:\Users\denhal.WISMAIN\Weizmann Institute Dropbox\DeanAdam Halperin\BrainValue\Code\neural_pipeline"))
# Now you can import the class
importlib.reload(ripple_io.ripple_loader)
from ripple_io.ripple_loader import RippleLoader
from processing.preprocessing import DataPreProcessing
file_path = "C:/Users/denhal.WISMAIN/Desktop/20220228/datafile(1).ns5"

loader = RippleLoader(wanted_streams=['lfp','spikes'],proc_single=False, wanted_electrodes='all',event_port=1)
neural_data,event_data = loader.load(file_path)

data_processing = DataPreProcessing(neural_data=neural_data,event_data=event_data)
def run_pipeline(file_path, use_filter=False):
    loader = RippleLoader()
    data = loader.load(file_path)

    events = data['events']
    streams = data['neural_streams']

    if use_filter:
        streams = custom_filter(streams, 300, 3000, fs=30000)

    # Feature extraction could go here
    X, y = build_features_and_labels(events, streams)

    clf = EventClassifier()
    clf.train(X, y)
    preds = clf.predict(X)
    return preds
