from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np


def get_permissions():
    os.chdir("test")	#the directory that will contain APKs and the data.csv (change the name if you need)
    print("running apktool")
    perm_list = []
    name_list = []
    help_list = ['perm: SET_DEBUG_APP','perm: READ_PHONE_STATE','perm: RECORD_AUDIO','perm: INTERNET','perm: PROCESS_OUTGOING_CALLS','perm: ACCESS_FINE_LOCATION','perm: RECEIVE_BOOT_COMPLETED',
        'perm: ACCESS_COARSE_LOCATION','perm: RECEIVE_SMS','perm: WRITE_SMS','perm: SEND_SMS','perm: UNINSTALL_SHORTCUT','perm: INSTALL_SHORTCUT','perm: SET_PREFERRED_APPLICATIONS']
    files = []
    for apk in Path('.').glob("*.apk"):
        print("working on", apk)
        files.append(str(apk)[0:len(str(apk))-4])
        os.system("apktool d -f "+str(apk))
        name_list.append(str(apk))

    print(files)
    new_f = open("perm.txt", "a")
    df = pd.DataFrame(help_list)
    first = 0
    for fileManifest in files:
        print("working of manifest for "+fileManifest)
        f = open(fileManifest+"/AndroidManifest.xml", "r")
        count = 0
        perm_list = []
        for line in f:
            x = re.search('<uses-permission android:name="*"', line)
            if x:
                count = count + 1
                apk_perms = pd.Series(line)
                apk_perms = apk_perms.apply(lambda perm: "perm:" + perm[19:])
                perm_list.append(' '.join(list(apk_perms)))
	    	
        list_of_names = [fileManifest+".apk"]*count
        tfidf = CountVectorizer(lowercase=False)
        if len(perm_list):
                print(perm_list)
                temp_list = tfidf.fit_transform(perm_list)
                dfapp = pd.DataFrame.sparse.from_spmatrix(
		    temp_list, index=list_of_names, columns="perm: " + tfidf.get_feature_names_out())
                dfapp.drop("perm: perm", axis=1, inplace=True)
                dfapp.drop("perm: name", axis=1, inplace=True)
                dfapp.drop("perm: android", axis=1, inplace=True)
		# dfapp.to_csv(str(fileManifest)+"our_data.csv")
                dfapp = dfapp.fillna(0)
                if first == 0:
                        first = 1
                        df = dfapp
                else:
                        df = df.combine_first(dfapp)
                        df = df.fillna(0)
    df = df.assign(type=None)
    df = df.assign(group_num=None)
    df = df.assign(group_mani=None)
    df = df.assign(category=None)
    df = df.assign(perm_rate=None)
    df = df.assign(call__=None)
    df = df.fillna(0)
    df.to_csv("1.csv")

def fix():
    help_list = ['perm: SET_DEBUG_APP','perm: READ_PHONE_STATE','perm: RECORD_AUDIO','perm: INTERNET','perm: PROCESS_OUTGOING_CALLS','perm: ACCESS_FINE_LOCATION','perm: RECEIVE_BOOT_COMPLETED',
        'perm: ACCESS_COARSE_LOCATION','perm: RECEIVE_SMS','perm: WRITE_SMS','perm: SEND_SMS','perm: UNINSTALL_SHORTCUT','perm: INSTALL_SHORTCUT','perm: SET_PREFERRED_APPLICATIONS']
    df = pd.read_csv('1.csv')
    for perm in help_list:
    	df[perm] = 0
    df.to_csv('data.csv')	

get_permissions()
#print("got premissions")
fix()
#print("fixed, need to label the data now")

'''
for now, you need to open the csv file that 'fix' made and change:
	1. change the second coloum name to 'name'
	2. add labeles to the 'tag' coloun (0/1)(benign or mallware)
	3. add labeles to the 'category' coloun (0/1/2/3)(train / test)
'''
