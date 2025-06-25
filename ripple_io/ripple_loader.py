# ripple_loader.py
'''
RippleLoader: A class to load and process Ripple data files (.ns5, .nev)
this class uses Ripple pyns library to read Ripple data files 
possibly works for BlackRock too
'''
import os
import glob
import numpy as np
import re
import pyns
from pyns.nsentity import EntityType as EntityType
from pyns.nsfile import NSFile as NSFile
import struct

class RippleLoader:
    def __init__(self, wanted_streams=['lfp','hi-res','spikes'],proc_single=False, wanted_electrodes='all',event_port=1,spike_units=2,timestamp_units_in_seconds=1):
        '''
        parameters:
        wanted_streams (list): list of streams to load (e.g.,['lfp','hi-res','raw','spikes'])
        proc_single (bool): if True, only load the .ns5 file without processing .nev files
        wanted_electrodes (list): list of electrodes to load (e.g., ['all', '1', '2', ...])
        event_port (int): the port number for events, default is 1
        spike_units (int): 0 - threshold crossing, 1 - sorted, 2 - both
        timestamp_units_in_seconds(int): timestamp resolution 1 - second, 1000 - ms
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
        ################################
        #TODO: find out the stream name for hires and fix 
        if 'hires' in wanted_streams or 'sbp' in wanted_streams:
            hires_streams = wanted_streams.index('hires') or  wanted_streams.index('sbp')
            wanted_streams[hires_streams] = 'hi-res'
        #############################

        self.wanted_streams = wanted_streams
        self.proc_single = proc_single
        self.wanted_electrodes = wanted_electrodes
        self.event_port = event_port
        self.spike_units = spike_units
        self.timestamp_res = timestamp_units_in_seconds

    def _find_files(self, base_path):
        # Check for file and numbered segments
        base_no_ext = os.path.splitext(base_path)[0]
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_no_ext)

        ns5_pattern = os.path.join(base_dir, f"{base_name}*.ns5")
        return sorted(glob.glob(ns5_pattern))

    def load(self, file_path,base_file_name='datafile'):
        '''
        load ripple data from all .ns5 files in the directory. 
        if proc_single is false, load also associated .nev files
        file_path (str): .ns5 full path
        base_file_name (str): base name of the file, used to find all segments

        '''
        ns5_files = self._find_files(file_path)
        if not ns5_files:
            raise FileNotFoundError(f"No .nsx files found for {file_path}")
        
        neural_data = {}
        event_data = {'label': [], 'timestamp': []}
        event_timestamp = []
        event_label = []

        # create a regex pattern to match the base file name, to handle segments
        file_name_pattern = re.escape(base_file_name) + r"(?:\((\d+)\))?" 

        # iterate over all files (segments) to concatanate data
        for fpath in ns5_files:

            match = re.search(file_name_pattern, fpath)
            if match.group(1): # if there was a segment number within (#) in the file name
                segment_number = int(match.group(1)) - 1 # to start from 0
            else:
                # if no segment number, set to None
                segment_number = None

            
            nsfile = NSFile(file_path,proc_single=self.proc_single)

            # Get analog and spike signals first:
            if self.wanted_electrodes == 'all':
                # no need to filter based on electrode, only stream type
                wanted_entities = [
                e for e in (
                list(nsfile.get_entities(EntityType.analog)) +
                list(nsfile.get_entities(EntityType.neural)))
                if (str_match := re.search(r'\D+', e.label))        
                and str_match.group().strip() in self.wanted_streams
                ]
            else:
                # Filter entities based on wanted electrodes
                # this requires extracting the entitiy.label, finding the number within the string and checking if it is in the wanted_electrode list
                # and also checking the stream type
                wanted_entities = [
                e for e in (
                list(nsfile.get_entities(EntityType.analog)) +
                list(nsfile.get_entities(EntityType.neural)))
                if  (str_match := re.search(r'\D+', e.label))
                and e.electrode_id in self.wanted_electrodes
                and str_match.group().strip() in self.wanted_streams
                ]
            
            if not wanted_entities:
                raise ValueError(f"No entities found for the specified streams: {self.wanted_streams} and electrodes: {self.wanted_electrodes}")

            # iterate over entities and feed to designated method
            for entity in wanted_entities:
                # extract data from entity according to stream type
                lfp,hires,raw,spikes,lfp_fs,hires_fs,raw_fs = self._load_entity_data(entity)
                electrode = entity.electrode_id


                # initilize or add data to the neural_data dictionary according to stream type
                if lfp is not None:
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'lfp', new_data = lfp,fs=lfp_fs)
                if hires is not None:
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'hi-res', new_data = hires,fs=hires_fs)
                if raw is not None:
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'raw', new_data = raw,fs=raw_fs)
                if spikes is not None:
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'spikes', spike_timestamps=spikes['timestamps']*self.timestamp_res,spike_ids=spikes['id'])

            # Events (digital markers)
            event_entity = nsfile.get_entities(EntityType.event)
            if event_entity:
                for i,entity in enumerate(event_entity):
                    if i ==  self.event_port:
                        event_num = entity.item_count
                        for event in range(event_num):
                            # get event timestamp
                            event_timestamp.append(entity.get_event_data(event)[0])
                            # get event number label
                            event_label.append(self.tuple_to_number_struct(entity.get_event_data(event)[1]))
            event_data = {'label': event_label, 'timestamp': event_timestamp*self.timestamp_res}
        return neural_data,event_data
    
    def _load_entity_data(self, entity):
        """Loads signal data from a single entity, based on its stream type."""
        stream_match = re.search(r'\D+', entity.label)
        spikes = {'id': [], 'timestamps' : []}
        if not stream_match:
            return None, None, None,None, None, None, None
        
        stream_type = stream_match.group().strip()
        electrode = int(re.search(r'\d+', entity.label).group())
        lfp,hires,raw,spikes,lfp_fs,hires_fs,raw_fs = None,None,None,None, None, None, None
        # Customize per stream
        if stream_type == 'lfp':
            lfp = entity.get_analog_data()
            lfp_fs = entity.sample_freq
        elif stream_type == 'hi-res':
            hires = entity.get_analog_data()
            hires_fs = entity.sample_freq
        elif stream_type == 'raw':
            raw = entity.get_analog_data()
            raw_fs = entity.sample_freq
        elif stream_type == 'spikes' or stream_type == 'elec':
            spike_num = entity.item_count
            ids = []
            timestamps = []
            for s in range(spike_num):
                ts,_,unit_id = entity.segment_entity.get_segment_data(s) # second returned value is unit waveform
                ids.append(unit_id)
                timestamps.append(ts)

            if self.spike_units == 0:
                # take only threshold crossing unsorted
                timestamps = timestamps[ids == 0]
                ids = ids[ids == 0]
            elif self.spike_units == 1:
                # take only sorted
                timestamps = timestamps[ids > 0 ]
                ids = ids[ids > 0]
            
            #else take no filtering, take both sorted and threhsold crossing
            spikes = {'id': ids, 'timestamps': timestamps}
        else:
            print(f"Unknown stream type: {stream_type} for entity {entity.label}")
            return None, None, None, None, None, None, None

        return lfp,hires,raw,spikes,lfp_fs,hires_fs,raw_fs

    def tuple_to_number_struct(self,signed_tuple):
        # ths converts the information from the digital events 

        # Convert each signed 16-bit int to bytes (little-endian, 2 bytes each)
        raw_bytes = b''.join(struct.pack('<h', val) for val in signed_tuple)
        
        # Use only first 4 bytes to get 32-bit unsigned integer (you expected 50000)
        number = struct.unpack('<I', raw_bytes[:4])[0]
        return number

    def add_data(self,neural_data, electrode, stream_type, new_data=None,spike_timestamps=None,spike_ids=None,fs=30000):
        '''
        the functions adds data into the neural_data dictionary, if the electrode and stream exist, is concatenates the new data to the existing data.
        if the electrode does not exist, it initializes it with the new data.

        Modifies neural_data in-place.

        Parameters:
        neural_data (dict): dictionary containing neural data
        electrode (int): electrode number
        stream_type (str): stream type
        new_data (np.ndarray): new data to be added, should be a numpy array
        spike_timestamps (np.ndarray): timestamps for spikes, should be a numpy array
        spike_ids (np.ndarray): ids for spikes, should be a numpy array
        fs (int): sampling rate (Hz)
        '''
        # Initialize electrode if it doesn't exist
        if electrode not in neural_data:
            neural_data[electrode] = {}
        

        # Add or concatenate data
        if stream_type in neural_data[electrode]:
                if stream_type == 'spikes':
                    neural_data[electrode][stream_type]['id'] = np.concatenate([
                    neural_data[electrode][stream_type]['id'],
                        spike_ids
                    ])

                    neural_data[electrode][stream_type]['timestamps'] = np.concatenate([
                        neural_data[electrode][stream_type]['timestamps'],
                        spike_timestamps
                    ])
                else:
                    neural_data[electrode][stream_type]['data'] = np.concatenate([
                    neural_data[electrode][stream_type]['data'],
                    new_data
                ])
        else:
            if  stream_type == 'spikes':
                neural_data[electrode][stream_type] = {
                    'id': spike_ids,
                    'timestamps': spike_timestamps
                }
            else:
                neural_data[electrode][stream_type] = {'data': new_data,'fs':fs}
''''
# Example usage:
from io.ripple_loader import RippleLoader

loader = RippleLoader()
data = loader.load("path/to/datafile.ns5")

print("Signals:", data['signals'].keys())
print("Events:", data['event_timestamps'].shape)
print("Spikes:", len(data['spikes']))
'''