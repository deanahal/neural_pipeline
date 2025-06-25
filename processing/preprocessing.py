import numpy as np
class DataPreProcessing:
    def __init__(self,neural_data,event_data,timestamp_units_in_seconds=1):
        self.neural_data = neural_data
        self.event_data = event_data

        # timestamp resolution, 1 - second 1000-ms
        self.timestamp_units_in_seconds = timestamp_units_in_seconds
        self.neural_data_locked = {}
        self.event_ts = {}
    def extract_event_times(self):
        '''
        Extract and store timestamps for each requested event label.
        timestampes returned at ms resolution
       
        '''
        if isinstance(self.event_ids,int):
            self.event_ids = [self.event_ids]
        
        # for every wanted event_id, extract all time stamps
        for i,id in enumerate(self.event_ids):
            labels = np.array(self.event_data['label'])
            timestamps = np.array(self.event_data['timestamp'])
            # filter events by start-end timestamps and event labels
            mask = (labels == id) & (timestamps >= self.start_ts) & (timestamps <= self.end_ts | self.end_ts == 0)

            if self.timestamp_units_in_seconds == 1:
                seconds_to_ms = 1000
            else:
                seconds_to_ms = 1
            self.event_ts[i] = {'event name': self.event_names[i], 'timestamps': np.round(timestamps[mask]*seconds_to_ms)}
    def lock_data_to_event(self,event_ids,event_names,time_before_event=1000,time_after_event=1000,start_ts=0,end_ts=0,bin_data=None):
        '''
        lock data around events

        event_ids (list): event ids(labels) to use
        time_before_event (int) - number of samples to lock before event (ms)
        time_after_event (int) - number of samples to lock after event (ms)
        timestamp_units_in_seconds (int) - timestamp resoultion - 1 - second, 1000 - ms
        start_ts and end_ts (int) - min and max timestamps of events. if end_ts = 0, no max limit
        bin_size (int): bin size in ms - if None - no bins
        '''
        self.event_ids = event_ids
        self.event_names = event_names
        self.time_before_event = time_before_event
        self.time_after_event = time_after_event
        self.start_ts = start_ts
        self.end_ts = end_ts

        # to update self.event_ts
        self.extract_event_times()

        for electrode in self.neural_data:
            streams = self.neural_data[electrode].keys()
            for stream in streams:
                if stream == 'spikes':
                    self.neural_data[electrode][stream]['timestamps']
                    ############### 
                    #TODO function to lock spiking data
                else:
                    # lock analog data
                    sample_rate = self.neural_data[electrode][stream]['fs']
                    data = self.neural_data[electrode][stream]['data']

                    # iterate over events and extract data
                    for event_number in range(len(self.event_ts)):
                        self.neural_data_locked[electrode][stream][event_number] = self.lock_analog_data(data,sample_rate,self.event_ts[event_number]['timestamps'])
    def lock_analog_data(self,data,fs,event_timestamps):
        '''
        for every electrode, and every stream, extract X seconds of data around each event
        data (numpy array): analog data
        fs (int): sampling rate
        event_timestamps (np.array) timestamps of current event
        '''

        time_before_ms = self.time_before_event
        time_after_ms = self.time_after_event
        ms_to_samples_ratio = fs/1000  # if sample rate is 2000 (hz) then for every ms there are 2 samples
        time_before_in_samples = time_before_ms*ms_to_samples_ratio
        time_after_in_samples = time_after_ms*ms_to_samples_ratio

        # take the relevant chunk of data around each timestamp
        locked = [
            data[i - time_before_in_samples:i + time_after_in_samples]
            for i in event_timestamps
            if i - time_before_in_samples >= 0 and i + time_after_in_samples <= len(data)
        ]

        if self.bin_size is not None:
            locked_and_bined = self.bin_data(locked)
        
        return locked_and_bined

    
    def bin_data(self):
        pass
    def normalize_data(self):
        pass