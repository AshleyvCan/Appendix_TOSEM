import pynvml
import threading
import csv
import time
from datetime import datetime


""" 
reused and modified from: 
Alizadeh, N., Belchev, B., Saurabh, N., Kelbert, P., & Castor, F. (2025, April). 
Language Models in Software Development Tasks: An Experimental Analysis of Energy and Accuracy. 
In 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR) (pp. 725-736). IEEE.
"""

class GPURecorder:
    def __init__(self, filename, batch_size, measurement_interval):
        self.filename = filename
        self.batch_size = batch_size
        self.measurement_interval = measurement_interval
        self.stop_event = threading.Event()
        self.records_list = [['Batch', 
                              'power (W)', 
                              'pState',
                              'gpu_utilization', 
                              'memory_utilization',
                               'timestamp_utc',    
                              'elapsed_s']]
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.t0 = None
        self.mW_to_W = 1000

    def get_gpu_power(self):
        if not self.stop_event.is_set():
            threading.Timer(self.measurement_interval, self.get_gpu_power).start()
            power_draw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / self.mW_to_W
            pstat = pynvml.nvmlDeviceGetPowerState(self.gpu_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

            time_stamp = datetime.now().isoformat(timespec='milliseconds')
            elapsed = time.perf_counter() - self.t0 if self.t0 else 0.0

            self.records_list.append([self.batch_size, power_draw, pstat, util.gpu, util.memory, time_stamp, round(elapsed, 6),])

    def start(self):
        self.t0 = time.perf_counter() 

        self.get_gpu_power()

    def stop(self):
        self.stop_event.set()
        pynvml.nvmlShutdown()
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for record in self.records_list:
                writer.writerow(record)