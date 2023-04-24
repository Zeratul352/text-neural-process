import os
import csv
from sys import argv
import dvc.api

#csv_name = './datasets/manifest_scripted.csv'
#directory = './datasets/train_parts'
params = dvc.api.params_show()
directory =  params['create_manifest']['directory']  #argv[1]
csv_name = params['create_manifest']['csv_name']#argv[2]
files = os.listdir(directory)
header = ['number', 'filename']

manifest_data = []
for i in range(len(files)):
    manifest_data.append([i, directory + '/' + files[i]])
#set_parts = filter(lambda x: x.endswith('.conllu'), files)
#print(set_parts)
with open(csv_name, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(manifest_data)