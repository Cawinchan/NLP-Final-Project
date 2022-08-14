import sys
import yaml
import sklearn_crfsuite
from pathlib import Path
import numpy as np 
import os
import conlleval
from yaml import Loader, Dumper
import string

train_dir = Path(__file__).resolve().parent/"data"/"train"
devin_dir = Path(__file__).resolve().parent/"data"/"dev.in"
test_dir = Path(__file__).resolve().parent/"data"/"test.in"
path_out = Path(__file__).resolve().parent/"data"/"convert_out_dev.txt"

START_STATE_KEY = "START"
STOP_STATE_KEY = "STOP"

def tokenize(file_path):  
    data, lst = [], []
    with open(file_path, 'r') as f:  
        for line in f:
            if line== '\n':
                data.append(lst)
                lst = []    
            else:
                lines = line.replace("\n",'').split(" ")
                lst.append(tuple(lines))
    return data

dicto = {
    'B-positive':'B-POS', 
    'B-negative': 'B-NEG',
    'B-neutral': 'B-NEU' , 
    'I-positive': 'I-POS', 
    'I-negative': 'I-NEG', 
    'I-neutral': 'I-NEU'
}

def converter(data, out_path):
    hashes = '####'
    for sentence in data:
        temp_sentence = ''
        temp_tagging = ''
        for word, tag in sentence:
            if word not in string.punctuation:
                temp_sentence = temp_sentence + " " + word
            else:
                temp_sentence += word

            temp_tagging += word
            temp_tagging += '='
            if tag in dicto:
                temp_tagging += dicto[tag]
            else:
                temp_tagging += tag

            temp_tagging += ' '
            line = temp_sentence+hashes+temp_tagging

        with open(out_path, "a+") as file_object:
            file_object.seek(0)
            data = file_object.read(100)

            if len(data) > 0 :
                file_object.write("\n")

            file_object.write(line)

    return 0 

def test_converter(data, out_path):
    hashes = '####'
    for sentence in data:
        line=''
        temp_sentence = ''
        temp_tagging = ''
        print(sentence)
        for word, tag in sentence:
            if word not in string.punctuation:
                temp_sentence = temp_sentence + " " + word
            else:
                temp_sentence += word

            temp_tagging += word
            temp_tagging += '='
            temp_tagging += 'O'

            temp_tagging += ' '
            line = temp_sentence+hashes+temp_tagging

        with open(out_path, "a+") as file_object:
            file_object.seek(0)
            data = file_object.read(100)

            if len(data) > 0 :
                if line != '':
                    file_object.write("\n")
            if line != '':
                file_object.write(line)

    return 0

#use test_converter for datasets without labels and converter for datasets with labels
if __name__ == "__main__":

    data_in = tokenize(test_dir)
    # converter(tokenize(test_dir) , path_out)
    test_converter(data_in, path_out)