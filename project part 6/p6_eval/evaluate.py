import numpy as np 
import os
import conlleval
from pathlib import Path

#simply fill in the prediction_dir and truth_dir and run the file to evaluate the predicted file
prediction_dir = Path(__file__).resolve().parent/"dataset"/"p6_ii_dev.out"
truth_dir = Path(__file__).resolve().parent/"dataset"/"dev.out"

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


def evaluate_results(truth_dir,prediction_dir):
    predictions = []
    prediction_sentences = tokenize(prediction_dir)
    for sentence in prediction_sentences:
        for word_pair in sentence:
            predictions.append(word_pair[1])     
    lines = """"""
    idx = 0
    with open(truth_dir, "r", encoding="utf8") as tstr:
        for line in tstr:
            if len(line) > 1:
                newline = line.replace("\n",f" {predictions[idx]}\n")
                lines += newline
                idx += 1
            else:
                lines += "\n"
    return lines.splitlines()

lines = evaluate_results(truth_dir,prediction_dir)
res = conlleval.evaluate(lines)
print(conlleval.report(res))