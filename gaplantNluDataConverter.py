#! /usr/bin/python3
# -*- coding: utf-8 -*-

import json
from pprint import pprint



with open('Dataset/gaplant/gaplant_nlu.json', encoding='utf-8') as data_file:
    data = json.load(data_file)

pprint(data) #data는 json 전체를 dictionary 형태로 저장하고 있음

#-----여기까지 동일-----

writeFile = open('Dataset/gaplant/data.json', encoding='utf-8',  mode='a')

jsonData = {}
for intentsIdx, intent in enumerate(data["intents"]):
    intentString = intent['intent']
    for patternIdx, pattern in enumerate(intent['patterns']):

        print(intentString[0], "::", pattern)

        jsonData['id'] = str(intentsIdx).zfill(4) + str(patternIdx).zfill(4)
        jsonData['source'] = pattern
        jsonData['target'] = intentString[0]

        print(jsonData)
        writeFile.writelines(str(jsonData).replace('\'', '\"') + '\n')
        # with open('Dataset/gaplant/data.json',  mode='a') as outfile:
        #     json.dump(data, outfile)

writeFile.close()