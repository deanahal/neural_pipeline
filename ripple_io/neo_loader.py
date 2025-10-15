# ripple_loader.py
'''
RippleLoader: A class to load and process Ripple data files (.ns5, .nev)
using neo library - loading data faster compared to Ripple pyns library, however, 
not all data streams are supported
'''
import os
import glob
import numpy as np
import re
import pyns
import neo
from neo import BlackrockIO
import struct


###############################################
#TODO - currently supports only spiking data
##############################################3
class NeoLoader:
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
       
       # change the names of streams to names that fit the stream labels
        if 'spikes' in wanted_streams or 'spike' in wanted_streams:
           spike_streams = wanted_streams.index('spikes') if 'spikes' in wanted_streams else wanted_streams.index('spike')
           wanted_streams[spike_streams] = 'elec'
       

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
        pattern = os.path.join(base_path, f"{base_file_name}*.ns5")

        return sorted(glob.glob(pattern))

    def load(self, file_path,base_file_name='datafile'):
        '''
        load ripple data from all .ns5 files in the directory. 
        if proc_single is false, load also associated .nev files
        file_path (str): .ns5 full path (without file name)
        base_file_name (str): base name of the file, used to find all segments

        '''
        # create a regex pattern to match the base file name, to handle segments
        file_name_pattern = re.escape(base_file_name) + r"(?:\((\d+)\))?" 
        ns5_files = self._find_files(file_path,base_file_name)
        if not ns5_files:
            raise FileNotFoundError(f"No .nsx files found for {file_path}")
        
        neural_data = {}
        event_timestamps = []
        event_labels = []
        analog_stream = []
        t_start_of_segment = None
        analog_event_data = None
        # iterate over all files (segments) to concatanate data
        # sort by file name (segment number)
        ns5_files = sorted(ns5_files,key=lambda s: int(re.search(r'\((\d+)\)', s).group(1)))
        for fpath in ns5_files:

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
            reader = BlackrockIO(filename = fpath)
            block = reader.read_block()
            for current_segment in block.segments:
                if t_start_of_segment is None:
                    t_start_of_segment = 0
                else:
                    # add length of previous segment to be the '0' start point of the current segment
                    t_start_of_segment = current_segment.t_stop.rescale('s').magnitude + t_start_of_segment
                
                
                #get spikes
                num_trains = len(current_segment.spiketrains)
                train_counter = 0
                # intilize spike dict 
                spikes = {'id': [], 'timestamps' : []}
                for st in current_segment.spiketrains:
                    
                    train_counter+=1
                    electrode = st.annotations.get('channel_id')
                    if (electrode in self.wanted_electrodes) or (self.wanted_electrodes == 'all'):
                        print(f"Unit: {st.annotations.get('unit_id', '?')}, Channel: {st.annotations.get('channel_id', '?')}")

                        unit_id = st.annotations.get('unit_id')
                        timestamps = st.times.rescale('s').magnitude + t_start_of_segment
                        
                        if timestamps.size > 0: # if there are any new spikes
                            self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'spikes', timestamps=timestamps*self.timestamp_res,unit_id=unit_id)




                # digital events
                if self.event_port is not None:
                    if self.blackrock == True:
                        
                        if current_segment.events[0].size > 0:
                            for event_port in current_segment.events:
                                if 'serial' in event_port.name:
                                    # save events from serial port
                                    event_labels.append(event_port.labels)
                                    event_timestamps.append(event_port.times.rescale('s').magnitude+t_start_of_segment)
                                    event_data = {'labels':event_labels,'timestamps':event_timestamps}
                else:
                    event_data = None
                # analog events
                if self.analog_channels is not None:
                    analog_channel = self.analog_channels[0]
                    signal = block.segments[0].analogsignals[2].magnitude[:,analog_channel]
                    analog_stream.append(signal)
                
                    

            if self.analog_channels is not None:
                analog_event_data = np.concatenate(analog_stream, axis=0)
            
        # Concatenate the lists into numpy arrays
        for ne in neural_data:
            for stream in neural_data[ne]:
                if stream == 'spikes':
                    for unit_id in neural_data[ne][stream]:
                        neural_data[ne][stream][unit_id]['timestamps'] = np.concatenate(neural_data[ne][stream][unit_id]['timestamps'], axis=0)
                else:
                    neural_data[ne][stream]['data'] = np.concatenate(neural_data[ne][stream]['data'], axis=0)
            
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
        

        # Add or concatenate data
        if stream_type in neural_data[electrode]:
                
                if stream_type == 'spikes':
                    if unit_id not in neural_data[electrode][stream_type]:
                        neural_data[electrode][stream_type][unit_id] ={'unit id': unit_id,'timestamps': []}
                       
                    neural_data[electrode][stream_type][unit_id]['timestamps'].append(timestamps)
                else:
                    neural_data[electrode][stream_type]['data'].append(new_data)
        else:
            if  stream_type == 'spikes':
                neural_data[electrode][stream_type] = {}
                neural_data[electrode][stream_type][unit_id] = {'unit id': unit_id,
                    'timestamps':[]}
                neural_data[electrode][stream_type][unit_id]['timestamps'].append(timestamps)
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