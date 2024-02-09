import os
import subprocess
import io
import time
import random
import string
import pandas as pd
from pynvml import *
from multiprocessing import Process, Queue, Event


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)


def power_loop(queue, event, interval):
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    device_list = [nvmlDeviceGetHandleByIndex(idx) for idx in range(device_count)]
    power_value_dict = {
        idx : [] for idx in range(device_count)
    }
    power_value_dict['timestamps'] = []
    last_timestamp = time.time()

    while not event.is_set():
        for idx,handle in enumerate(device_list):
            power = nvmlDeviceGetPowerUsage(handle)
            power_value_dict[idx].append(power*1e-3)
        timestamp = time.time()
        power_value_dict['timestamps'].append(timestamp)
        wait_for = max(0,1e-3*interval-(timestamp-last_timestamp))
        time.sleep(wait_for)
        last_timestamp = timestamp
    queue.put(power_value_dict)

class GetPower(object):
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()
        
        interval = 100 #ms
        self.smip = Process(target=power_loop,
                args=(self.power_queue, self.end_event, interval))
        self.smip.start()
        return self
    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)
    def energy(self):
        import numpy as np
        _energy = []
        energy_df = self.df.loc[:,self.df.columns != 'timestamps'].astype(float).multiply(self.df["timestamps"].diff(),axis="index")/3600
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy

if __name__ == "__main__":
    from utils import args_parser
    from Train import main
    args = args_parser()
    args.epochs = 3
    args.data_path = '/p/scratch/deepacf/ENS10_ERA5/netCDFs' #netCDF files
    args.model = 'UNet'
    
    with GetPower() as measured_scope:
        print('Measuring Energy during main() call')
        try:
            main(args)
        except Exception as exc:
            import traceback
            print(f"Errors occured during training: {exc}")
            print(f"Traceback: {traceback.format_exc()}")
    print("Energy data:")

    print  (measured_scope.df)
    pd_name = get_random_string(5)
    print(f"Saving energy data to {pd_name}.csv")
    measured_scope.df.to_csv(f"{pd_name}.csv")
    print("Energy-per-GPU-list:")
    energy_int = measured_scope.energy()
    print(f"integrated: {energy_int}")
