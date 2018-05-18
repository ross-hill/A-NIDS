#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This file contains a small amount of preprocessing before modelling.
    
    Use of this code applies under an Apache 2.0 licence.
"""

import pandas as pd
import numpy as np
import time

fdata = None # read_csv


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
for a in range(fdata.shape[0]):
    if not is_number(fdata.iloc[a,25]):
        dd = fdata.iloc[a,25]
        i1 = dd.find(".")
        i2 = dd[i1+1:].find(".")
        fdata.iloc[a,25] = dd[i1+1:][:i2]
fdata.iloc[:, 11] = fdata["duration"].astype(int)
fdata.iloc[:, 25] = fdata.iloc[:,25].astype(int)

all_times = fdata["timestamp"]
hours = np.zeros(len(all_times))
for t,i in zip(all_times, range(len(all_times))):    
    hour = time.gmtime(np.array(t)).tm_hour
    hours[i] = hour
fdata["hour"] = hours

fdata = fdata.drop(["timestamp"], axis = 1)
fdata_dumm = fdata.drop(["dst_port", "duration","recv_bytes", "count", "sent_bytes", "sent_pkts", "src_port", "transmittedkbits"], axis = 1)
fdata_ndumm = fdata.loc[:,["dst_port", "duration","recv_bytes", "count", "sent_bytes", "sent_pkts", "src_port", "transmittedkbits"]]
fdata_dumm = pd.get_dummies(fdata_dumm, drop_first=True)
fdata_dumm = fdata_dumm.drop(["hour"], axis = 1)
hour_dumm = pd.get_dummies(fdata["hour"], prefix="hour", drop_first = True)
fdata_fit = pd.concat([fdata_dumm, fdata_ndumm, hour_dumm], axis = 1)