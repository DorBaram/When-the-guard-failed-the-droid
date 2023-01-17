from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np


def fix():
    help_list = ['perm: SET_DEBUG_APP','perm: READ_PHONE_STATE','perm: RECORD_AUDIO','perm: INTERNET','perm: PROCESS_OUTGOING_CALLS','perm: ACCESS_FINE_LOCATION','perm: RECEIVE_BOOT_COMPLETED',
        'perm: ACCESS_COARSE_LOCATION','perm: RECEIVE_SMS','perm: WRITE_SMS','perm: SEND_SMS','perm: UNINSTALL_SHORTCUT','perm: INSTALL_SHORTCUT','perm: SET_PREFERRED_APPLICATIONS']
    df = pd.read_csv('1.csv')
    for perm in help_list:
    	df[perm] = 0
    df.to_csv('data.csv')
fix()

print("Done\n")

