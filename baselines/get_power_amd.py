# Don't forget to
#    export PYTHONPATH=/opt/rocm/libexec/rocm_smi/:$PYTHONPATH

import os
import subprocess
import io
import time
import random
import string
import pandas as pd
from rsmiBindings import *
from multiprocessing import Process, Queue, Event

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)


def power_loop(queue, event, interval):
    ret = rocmsmi.rsmi_init(0)
    if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
        raise RuntimeError("Failed initializing rocm_smi library")
    device_count = c_uint32(0)
    ret = rocmsmi.rsmi_num_monitor_devices(byref(device_count))
    if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
        raise RuntimeError("Failed enumerating ROCm devices")
    device_list = list(range(device_count.value))
    power_value_dict = {
        id : [] for id in device_list
    }
    power_value_dict['timestamps'] = []
    last_timestamp = time.time()
    start_energy_list = []
    for id in device_list:
        energy = c_uint64()
        energy_timestamp = c_uint64()
        energy_resolution = c_float()
        ret = rocmsmi.rsmi_dev_energy_count_get(id, 
                byref(energy),
                byref(energy_resolution),
                byref(energy_timestamp))
        if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
            raise RuntimeError(f"Failed getting Power of device {id}")
        start_energy_list.append(round(energy.value*energy_resolution.value,2)) # unit is uJ

    while not event.is_set():
        for id in device_list:
            power = c_uint32()
            ret = rocmsmi.rsmi_dev_power_ave_get(id, 0, byref(power))
            if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
                raise RuntimeError(f"Failed getting Power of device {id}")
            power_value_dict[id].append(power.value*1e-6) # value is uW
        timestamp = time.time()
        power_value_dict['timestamps'].append(timestamp)
        wait_for = max(0,1e-3*interval-(timestamp-last_timestamp))
        time.sleep(wait_for)
        last_timestamp = timestamp

    energy_list = [0.0 for _ in device_list]
    for id in device_list:
        energy = c_uint64()
        energy_timestamp = c_uint64()
        energy_resolution = c_float()
        ret = rocmsmi.rsmi_dev_energy_count_get(id, 
                byref(energy),
                byref(energy_resolution),
                byref(energy_timestamp))
        if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
            raise RuntimeError(f"Failed getting Power of device {id}")
        energy_list[id] = round(energy.value*energy_resolution.value,2) - start_energy_list[id]

    energy_list = [ (energy*1e-6)/3600 for energy in energy_list] # convert uJ to Wh
    queue.put(power_value_dict)
    queue.put(energy_list)

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
        self.energy_list_counter = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)
    def energy(self):
        import numpy as np
        _energy = []
        energy_df = self.df.loc[:,self.df.columns != 'timestamps'].astype(float).multiply(self.df["timestamps"].diff(),axis="index")/3600
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy,self.energy_list_counter


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
    
    pd_name = get_random_string(5)+'_amd'
    print(f"Saving energy data to {pd_name}.csv")
    measured_scope.df.to_csv(f"{pd_name}.csv")
    
    print("Energy-per-GPU-list:")
    energy_int,energy_cnt = measured_scope.energy()
    print(f"integrated: {energy_int}")
    print(f"from counter: {energy_cnt}")
    # f.close()
