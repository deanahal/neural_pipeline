# ripple_loader.py
'''
BlackRockloader: A class to load and process Ripple data files (.ns5, .nev)
using brpy library -https://github.com/BlackrockNeurotech/Python-Utilities
loading data faster compared to Ripple pyns library, however, 
not all data streams are supported
'''
import os
import glob
import numpy as np
import re
import pyns
import sys
sys.path.append(os.path.abspath(r"C:\Users\denhal.WISMAIN\Weizmann Institute Dropbox\DeanAdam Halperin\BrainValue\Code"))

import brpylib
from brpylib import NevFile,NsxFile
import struct


###############################################
#TODO - currently supports only spiking data
##############################################3
class BlackRockloader:
    def __init__(self, wanted_streams=['spikes'], wanted_electrodes='all',event_port=None,analog_channels=None,spike_units=2,timestamp_units_in_seconds=1,first_segment=0,last_segment=999,blackrock=False):
        '''
        parameters:
        wanted_streams (list): list of streams to load (e.g.,['lfp','hi-res','raw','spikes'])
        wanted_electrodes (list): list of electrodes to load (e.g., ['all', '1', '2', ...])
        event_port (int): the port number for events, default is None
        spike_units (int): 0 - threshold crossing, 1 - sorted, 2 - both
        timestamp_units_in_seconds(int): timestamp resolution 1 - second, 1000 - ms
        analog_channels (list): list of analog channels to load deafult is None
        first_segment (int): first segment to load (default 0)
        last_segment (int): last segment to load (default 999)
        blackrock(boolean): if True - blackrock data - important for data decoding (data is saved through serial port)
        '''
        if wanted_streams == 'all':
            wanted_streams = ['lfp', 'hi-res', 'raw','spikes']
        allowed_streams = {'lfp', 'hires', 'hi-res', 'elec', 'raw', 'sbp','spikes','spike'}
        invalid_streams = [s for s in wanted_streams if s not in allowed_streams]
        if invalid_streams:
            raise ValueError(f"Invalid stream types found: {invalid_streams}. Allowed types are: {allowed_streams}")
       
  
       

        self.wanted_streams = wanted_streams
        self.wanted_electrodes = wanted_electrodes
        self.event_port = event_port
        self.spike_units = spike_units
        self.timestamp_res = timestamp_units_in_seconds
        self.analog_channels = analog_channels
        self.first_segment = first_segment
        self.last_segment = last_segment
        self.blackrock = blackrock
    def _find_files(self, base_path,base_file_name):
       # Build search pattern using the variable
        ns5_pattern = os.path.join(base_path, f"{base_file_name}*.ns5")
        nev_pattern = os.path.join(base_path, f"{base_file_name}*.nev")
        return sorted(glob.glob(ns5_pattern)),sorted(glob.glob(nev_pattern))

    def load(self, file_path,base_file_name='datafile'):
        '''
        load ripple data from all .ns5 files in the directory. 
        if proc_single is false, load also associated .nev files
        file_path (str): .ns5 full path (without file name)
        base_file_name (str): base name of the file, used to find all segments

        '''
        # create a regex pattern to match the base file name, to handle segments
        file_name_pattern = re.escape(base_file_name) + r"(?:\((\d+)\))?" 
        ns5_files,nev_files = self._find_files(file_path,base_file_name)
        if not ns5_files:
            raise FileNotFoundError(f"No .nsx files found for {file_path}")
        
        neural_data = {}
        event_timestamps = []
        event_labels = []
        first_segment = True
        t_start_of_segment = 0

        # sort nev files so that they are by the order of segments
        nev_files = sorted(nev_files, key=lambda s: int(re.search(r'\((\d+)\)', s).group(1)))
        ns5_files = sorted(ns5_files,key=lambda s: int(re.search(r'\((\d+)\)', s).group(1)))
        # iterate over all files (segments) to concatanate data
        for file_number,fpath in enumerate(nev_files):

            match = re.search(file_name_pattern, fpath)
            if match.group(1): # if there was a segment number within (#) in the file name
                segment_number = int(match.group(1)) - 1 # to start from 0
                print('processing sgement number'+str(segment_number))
                if segment_number < self.first_segment:
                    continue
                if segment_number > self.last_segment:
                    break
                
            else:
                # if no segment number, set to None
                segment_number = None
            nev = NevFile(fpath)
            nev_data = nev.getdata()

            if first_segment:
                # time of start of the first segment 
                time_origin = nev.basic_header['TimeOrigin']
                current_segment_start = 0
                first_segment = False
            else:
                # time from the start of the first segment
                delta_time = nev.basic_header['TimeOrigin'] - time_origin
                # convert delta time to seconds 
                current_segment_start = delta_time.total_seconds()


            if 'spikes' in self.wanted_streams:
                train_counter = 0
                electrodes = np.array(nev_data['spike_events']['Channel'])
                units = np.array(nev_data['spike_events']['Unit'])
                # add segment time start to the timestamp 
                timestamps = np.array(nev_data['spike_events']['TimeStamps'])/30000 + current_segment_start


                # intilize spike dict 
                spikes = {'id': [], 'timestamps' : []}
                ##########################3
                #TODO - it currently takes all units (sorted, unsorted) and does not differntiate
                #############################
                for electrode in np.unique(electrodes):
                    for u in np.unique(units[electrodes == electrode]):
                        train_counter+=1
                        unit_id = u
                        timestamps_current_unit = timestamps[(units == u) & (electrodes == electrode)]
                    
                        if timestamps_current_unit.size > 0: # if there are any new spikes
                            self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'spikes', timestamps=timestamps_current_unit*self.timestamp_res,unit_id=unit_id)
            


            if ('raw' in self.wanted_streams) or ('lfp' in self.wanted_streams) or ('hires' in self.wanted_streams) or ('hi-res' in self.wanted_streams):
                ns5 = NsxFile(ns5_files[file_number])
                extended_headers = ns5.extended_headers
                labels = [header['ElectrodeLabel'] for header in extended_headers]
                # indices of wanted stream and electrodes
                if (self.wanted_streams == 'all') or (self.wanted_electrodes == 'all'):
                    raise ValueError('specify streams and electrodes to load, all is not YET supported   ')
                else:
                    indices = [
                            i for i, e in enumerate(labels)
                            if (match_label := re.search(r'(\D+)\s*(\d+)', e))  # match 'raw 129'
                            and match_label.group(1).strip() in self.wanted_streams
                            and int(match_label.group(2)) in self.wanted_electrodes
                        ]
                if indices:
                    ns5_data = ns5.getdata()
                else:
                    raise ValueError('no valid stream and electrode comb. found')
                for i in indices:
                    stream_type = labels[i].split()[0].lower()
                    electrode = int(labels[i].split()[1])
                    # analog data corresponding to the stream and electrode:
                    data = np.array(ns5_data['data'][0][i,:])
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = stream_type,new_data=data)

                

            if self.event_port is not None:
                if self.blackrock == True:
                    event_timestamps = np.array(nev_data['digital_events']['TimeStamps'])/30000
                    event_labels = np.array(nev_data['digital_events']['UnparsedData'])
                    event_data = {'labels':event_labels,'timestamps':event_timestamps}
            else:
                event_data = None

            if self.analog_channels is not None:
                # TODO placeholder
                analog_event_data = 0
            else:
                analog_event_data = None
        
        # convert the list to a numpy array
        for electrode in neural_data:
            for stream_type in neural_data[electrode]:
                if stream_type != 'spikes':
                    neural_data[electrode][stream_type]['data'] = np.concatenate(neural_data[electrode][stream_type]['data'])
        return neural_data,analog_event_data,event_data
    def digital_event_decoder(self):
        '''
        decoding digital events to get label and timestamps
        return 
        labels
        timestamps
        '''
        1
    def analog_event_decoder(self,analog_signal):
        # decoding the analog signal to get event labels, and event timestamps
        # analog_signal (numpy array): the signal to decode
        analog_signal
    def tuple_to_number_struct(self,signed_tuple):
        # ths converts the information from the digital events 

        # Convert each signed 16-bit int to bytes (little-endian, 2 bytes each)
        raw_bytes = b''.join(struct.pack('<h', val) for val in signed_tuple)
        
        # Use only first 4 bytes to get 32-bit unsigned integer (you expected 50000)
        number = struct.unpack('<I', raw_bytes[:4])[0]
        return number

    def add_data(self,neural_data, electrode, stream_type, new_data=None,timestamps=None,unit_id=None,fs=30000):
        '''
        the functions adds data into the neural_data dictionary, if the electrode and stream exist, is concatenates the new data to the existing data.
        if the electrode does not exist, it initializes it with the new data.

        Modifies neural_data in-place.

        Parameters:
        neural_data (dict): dictionary containing neural data
        electrode (int): electrode number
        stream_type (str): stream type
        new_data (np.ndarray): new data to be added, should be a numpy array
        timestamps (np.ndarray): timestamps for spikes, should be a numpy array
        unit_id (np.ndarray): ids for spikes, should be a numpy array
        fs (int): sampling rate (Hz)
        '''
        # Initialize electrode if it doesn't exist
        if electrode not in neural_data:
            neural_data[electrode] = {}
        
        if stream_type == 'raw':
            fs = 30000
        # Add or concatenate data
        if stream_type in neural_data[electrode]:
                
                if stream_type == 'spikes':
                    if unit_id not in neural_data[electrode][stream_type]:
                        neural_data[electrode][stream_type][unit_id] ={'unit id': unit_id,'timestamps': timestamps}
                    else:
                       
                        neural_data[electrode][stream_type][unit_id]['timestamps'] = np.concatenate([
                            np.atleast_1d(neural_data[electrode][stream_type][unit_id]['timestamps']),
                            np.atleast_1d(timestamps)
                        ])
                else:
                    neural_data[electrode][stream_type]['data'].append(new_data) 
           
        else:
            if  stream_type == 'spikes':
                neural_data[electrode][stream_type] = {}
                neural_data[electrode][stream_type][unit_id] = {'unit id': unit_id,
                    'timestamps': timestamps
                }
            else:
                neural_data[electrode][stream_type] = {'data': [],'fs':fs}
                neural_data[electrode][stream_type]['data'].append(new_data) 
''''
# Example usage:
from io.ripple_loader import RippleLoader

loader = RippleLoader()
data = loader.load("path/to/datafile.ns5")

print("Signals:", data['signals'].keys())
print("Events:", data['event_timestamps'].shape)
print("Spikes:", len(data['spikes']))
'''