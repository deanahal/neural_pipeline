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
    def __init__(self, wanted_streams=['lfp','hi-res','spikes'],proc_single=False, wanted_electrodes='all',event_port=None,analog_channels=None,spike_units=2,timestamp_units_in_seconds=1,first_segment=0,last_segment=999,dead_electrode=[-1],load_audio=False,channels_in_zero_index=False):
        '''
        parameters:
        wanted_streams (list): list of streams to load (e.g.,['lfp','hi-res','raw','spikes'])
        proc_single (bool): if True, only load the .ns5 file without processing .nev files
        wanted_electrodes (list): list of electrodes to load (e.g., ['all', '1', '2', ...])
        event_port (int): the port number for events, default is 1
        spike_units (int): 0 - threshold crossing, 1 - sorted, 2 - both
        timestamp_units_in_seconds(int): timestamp resolution 1 - second, 1000 - ms
        analog_channels (list): list of analog channels to load
        first_segment (int): first segment to load (default 0)
        last_segment (int): last segment to load (default 999)
        dead_electrdoe (list): electrodes to remove default [-1]
        load_audio (boolean): load audio from analog channel (channel number 30) or not, default False
        channels_in_zero_index (boolean): if True, electrode channel numbers start from 0 (like online) - in this case input parameters should also be zero-indexes 
            if False, start from 1 (like in the files) and input parameters should also be one-indexed, default False.
            relevant for wanted_electrodes and dead_electrode parameters
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
      
        if 'hires' in wanted_streams or 'sbp' in wanted_streams:
            hires_streams = wanted_streams.index('hires') or  wanted_streams.index('sbp')
            wanted_streams[hires_streams] = 'hi-res'

        self.wanted_streams = wanted_streams
        self.proc_single = proc_single
        # if i want zero index, i need to fix the wanted_electrodes list to match the file indexing
        if channels_in_zero_index:
            if wanted_electrodes != 'all':
                wanted_electrodes = [ch + 1 for ch in wanted_electrodes]
        self.wanted_electrodes = wanted_electrodes
        self.event_port = event_port
        self.spike_units = spike_units
        self.timestamp_res = timestamp_units_in_seconds
        self.analog_channels = analog_channels
        self.first_segment = first_segment
        self.last_segment = last_segment
        # dead electrode is handeled after fixing indexing, no need to fix here
        self.dead_electrode = dead_electrode
        self.micro_electrode_ids = []
        self.macro_electrode_ids = []
        self.load_audio = load_audio
        self.channels_in_zero_index = channels_in_zero_index
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
        
        if len(ns5_files) >1:
            # sort files by name (segment)
            ns5_files = sorted(ns5_files,key=lambda s: int(re.search(r'\((\d+)\)', s).group(1)))
        this_segment_time_zero = None
        if not ns5_files:
            raise FileNotFoundError(f"No .nsx files found for {file_path}")
        
        neural_data = {}
        event_data = {'label': [], 'timestamp': []}
        event_timestamp = []
        event_label = []
        analog_event_data = {}
        analog_data = {}
        audio = []
        # iterate over all files (segments) to concatanate data
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

            
            nsfile = NSFile(fpath,proc_single=self.proc_single)
            time_span = nsfile.get_file_data('ns5').time_span
            if this_segment_time_zero is None:
                this_segment_time_zero = 0
                previous_time_span = time_span
            else:
                this_segment_time_zero = this_segment_time_zero + previous_time_span
                previous_time_span = time_span

            # if no electrode id saved, go to function to get the electrode ids of micro and macro
            if (not self.micro_electrode_ids) and (not self.macro_electrode_ids):
                self.micro_electrode_ids,self.macro_electrode_ids = self.micro_macro_ids(nsfile)

            # Get analog and spike signals first:
            if np.all(self.wanted_electrodes == 'all'):
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
                electrode = entity.electrode_id - (1 if self.channels_in_zero_index else 0) # to have electrode numbers starting from 0 (like online) or from 1 (like in the files)

                if electrode in self.dead_electrode:
                    continue
                    
                # extract data from entity according to stream type
                lfp,hires,raw,spikes,lfp_fs,hires_fs,raw_fs = self._load_entity_data(entity,this_segment_time_zero)
                if electrode in self.micro_electrode_ids:
                    electrode_type = 'micro'
                elif electrode in self.macro_electrode_ids:
                    electrode_type = 'macro'
                else:
                    electrode_type = 'unknown'

                # initilize or add data to the neural_data dictionary according to stream type
                if lfp is not None:
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'lfp', new_data = lfp,fs=lfp_fs,electrode_type=electrode_type)
                if hires is not None:
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'hi-res', new_data = hires,fs=hires_fs,electrode_type=electrode_type)
                if raw is not None:
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'raw', new_data = raw,fs=raw_fs,electrode_type=electrode_type)
                if spikes is not None:
                    self.add_data(neural_data = neural_data, electrode = electrode, stream_type = 'spikes', spike_timestamps=spikes['timestamps']*self.timestamp_res,spike_ids=spikes['id'],electrode_type=electrode_type)

            # Events (digital markers)
            event_entity = nsfile.get_entities(EntityType.event)
           
            if event_entity:
                for i,entity in enumerate(event_entity):
                    if i ==  self.event_port:
                        event_num = entity.item_count
                        for event in range(event_num):
                            # get event timestamp
                            event_timestamp.append(entity.get_event_data(event)[0] + this_segment_time_zero)
                            # get event number label
                            event_label.append(self.tuple_to_number_struct(entity.get_event_data(event)[1]))
                event_data = {'label': event_label, 'timestamp': event_timestamp*self.timestamp_res}

            ################################
            # get data from 'analog' entities. 
            #TODO differentiate between analog inputs that are event labels and other analog inputs
            #############################
            if self.analog_channels is not None:
                # take all relevant entities with label 'analog X' where X is a number from the analog_channels list
                wanted_entities = [
                e for e in nsfile.get_entities(EntityType.analog)
                if (match := re.fullmatch(r'analog (\d+)', e.label.strip()))
                and int(match.group(1)) in self.analog_channels
                ]            
                for entity in wanted_entities:
                    analog_ch = int(re.search(r'\d+', entity.label).group())
                    if analog_ch not in analog_data:
                        analog_data[analog_ch] = {'data':[]}

                    # every channel can have 2 different sampling rates - 1000Hz or 30kHz
                    # we take only the 30kHz one, since the 1000Hz is a downsampled version of the 30kHz
                    if entity.get_analog_info()[0] == 30000:
                        print(f"loading analog channel {analog_ch} with sampling rate {entity.get_analog_info()[0]}")
                        analog_data[analog_ch]['data'].append(entity.get_analog_data())
                    else:
                        print(f"Analog channel {analog_ch} has sampling rate {entity.get_analog_info()[0]}, expected 30000Hz, skipping entity")

            if self.load_audio:
                # take all relevant entities with label 'analog 30' which is audio. 
                audio_entities = [
                e for e in nsfile.get_entities(EntityType.analog)
                if e.label.strip().lower() == "analog 30"
                ]

                # there is one audio channel, but it can be save with 2 different sampling rates - 1000Hz or 30kHz
                # we take only the 30kHz one, since the 1000Hz is a downsampled version of the 30kHz
                for audio_entity in audio_entities:
                    audio_fs = audio_entity.get_analog_info()[0] # sample rate of audio either 1000hz, or 30khz
                    if audio_fs == 30000:
                        audio.append(audio_entity.get_analog_data())
                        print(f"loading audio channel")

                    else:
                        print(f"Audio channel has sampling rate {audio_fs}, expected 30000Hz, skipping entity")

                   
            
        # Concatenate the lists into numpy arrays
        for ne in neural_data:
            for stream in neural_data[ne]:
                if stream == 'spikes':
                    neural_data[ne][stream]['timestamps'] = np.concatenate(neural_data[ne][stream]['timestamps'], axis=0)
                    neural_data[ne][stream]['id'] = np.concatenate(neural_data[ne][stream]['id'], axis=0)
                else:
                    neural_data[ne][stream]['data'] = np.concatenate(neural_data[ne][stream]['data'], axis=0)
        # convert the analog data into event data
        for a_ch in analog_data:
            analog_data[a_ch]['data'] = np.concatenate(analog_data[a_ch]['data'],axis=0)

            # this is analog channel in 30khz fs, so in times - i convert samples into 1s scale timestamps
            analog_event_data[a_ch] = self.decode_with_timestamps(analog_data[a_ch]['data'],times = np.arange(len(analog_data[a_ch]['data']))/30000)

        # convert audio data into a single numpy array
        if self.load_audio:
            audio_data = {'data':np.concatenate(audio,axis=0),'fs':audio_fs}
        else:
            audio_data = {}
       
        return neural_data,analog_event_data,event_data,audio_data
    
    def _load_entity_data(self, entity,this_segment_time_zero):
        """Loads signal data from a single entity, based on its stream type."""
        stream_match = re.search(r'\D+', entity.label)
        spikes = {'id': [], 'timestamps' : []}
        if not stream_match:
            return None, None, None,None, None, None, None
        
        stream_type = stream_match.group().strip()      
        electrode = int(re.search(r'\d+', entity.label).group())
        print(entity.label)
        lfp,hires,raw,spikes,lfp_fs,hires_fs,raw_fs = None,None,None,None, None, None, None
        # Customize per stream
        if stream_type == 'lfp':
            lfp = entity.get_analog_data().astype('float32')
            lfp_fs = entity.sample_freq
        elif stream_type == 'hi-res':
            hires = entity.get_analog_data().astype('float32')
            hires_fs = entity.sample_freq
        elif stream_type == 'raw':
            raw = entity.get_analog_data().astype('float32')
            raw_fs = entity.sample_freq
        elif stream_type == 'spikes' or stream_type == 'elec':  
            spike_num = entity.item_count
            ids = []
            timestamps = []
            for s in range(spike_num):
                ts,_,unit_id = entity.segment_entity.get_segment_data(s) # second returned value is unit waveform
                ids.append(unit_id)
                timestamps.append(ts+this_segment_time_zero)

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

    def add_data(self,neural_data, electrode, stream_type, new_data=None,spike_timestamps=None,spike_ids=None,fs=30000,electrode_type='unkown'):
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
        electrode_type (str): type of the electrode ('micro', 'macro', 'unknown')
        '''
        # Initialize electrode if it doesn't exist
        if electrode not in neural_data:
            neural_data[electrode] = {}
        

        # Add or concatenate data
        if stream_type not in neural_data[electrode]:
                if  stream_type == 'spikes':
                    neural_data[electrode][stream_type] = {
                        'id': [],
                        'timestamps':[],
                    }
                
                else:
                    neural_data[electrode][stream_type] = {'data': [],'fs':fs,'electrode_type':electrode_type}
        if stream_type == 'spikes':
            neural_data[electrode][stream_type]['id'].append(spike_ids)
            neural_data[electrode][stream_type]['timestamps'].append(spike_timestamps)
            
        else:
            neural_data[electrode][stream_type]['data'].append(new_data)
    
            

    def micro_macro_ids(self,nsfile):
        """
        Extracts micro and macro electrode IDs 
        use the first file, read it as ns5, and read it as nf6 to get IDS
        notice - in order to read nf6 file, the nsfile.py file (pyns library) has to be modified - include in line 108: nsx_files += glob(filename[:-4] + '.nf6')

        parameters: 
        nsfile
        returns: tuple of two lists - micro and macro electrode ids
        """
        micro_electrode_ids = []
        macro_electrode_ids = []

        micro_data = nsfile.get_file_data('ns5')
        if micro_data is None:
            micro_electrode_ids = [] 
        else:
            for h in micro_data.parser.get_extended_headers():
                micro_electrode_ids.append(h.electrode_id)
        macro_data = nsfile.get_file_data('nf6')
        if macro_data is None:
            macro_electrode_ids = []
        else:
            for h in macro_data.parser.get_extended_headers():
                macro_electrode_ids.append(h.electrode_id)
        return sorted(micro_electrode_ids), sorted(macro_electrode_ids)
     
    def decode_with_timestamps(self,
        serial_data_mv, times,
        sample_rate=30000, bit_samples=6, threshold_mv=2500
    ):
        # Boaz's function decoding serial data from the analog port into bytes (labels), and correspondiing timestmaps
        logic = (serial_data_mv >= threshold_mv).astype(int)
        events = []
        i = 0
        N = len(logic)
        frame_len = bit_samples * (1 + 8 + 1 + 1)

        while i + frame_len <= N:
            if logic[i] != 0:
                i += 1
                continue
            
            start_time = times[i]
            bits = []
            for b in range(1 + 8 + 1 + 1):
                corr = 2 if b > 8 else 1 if b > 4 else 0
                sample_index = i + b * bit_samples + bit_samples // 2 + corr
                bits.append(logic[sample_index])

            start, data_bits, parity, stop = bits[0], bits[1:9], bits[9], bits[10]
            if start != 0:
                i += 1
                continue

            byte = sum(bit << idx for idx, bit in enumerate(data_bits))
            events.append({'time_s': start_time, 'byte': byte})
            i += frame_len

        return events
