import sys
import yaml
import sklearn_crfsuite
from pathlib import Path
import numpy as np 
import os
import conlleval
from yaml import Loader, Dumper

train_dir = Path(__file__).resolve().parent/"dataset"/"train"
devin_dir = Path(__file__).resolve().parent/"dataset"/"dev.in"
test_dir = Path(__file__).resolve().parent/"dataset"/"test.in"
out_dir = Path(__file__).resolve().parent/'dataset'/"test.p6.CRF.out"

feature_dict = {'current': 
                ['bias', 'current_word.lower()', 'current_word[-5]', 'current_word[-4]', 'current_word[-3]', 'current_word[-2]', 'current_word.isupper()', 'current_word.isdigit()', 'current_word.islower()'], 
                'previous': 
                ['previous_word[-2]', 'previous_word.lower()', 'previous_word.isupper()'], 
                'next': 
                ['next_word[-2]', 'next_word.lower()', 'next_word.isupper()']
                }

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

train_sentences = tokenize(train_dir)
# print(train_sentences[:5])      

class CRF(object):
    def __init__(self):
        return

    def fit(self, data):
        x_train = [self.sent_features(i, feature_dict) for i in data]
        y_train = [self.sent_labels(i) for i in data] 
        self.crf_model = self.train_crf(x_train, y_train)
    
    def predict(self, test_file, file_out):

        # print(test_file[:5])
        list_x = []
        for sentence in test_file:
            temp_list = []
            for word in sentence:
                temp_list.append(word[0])

            word_arr_features = self.sent_features_test(temp_list, feature_dict)
            list_x.append(word_arr_features)
        y_predict = self.crf_model.predict(list_x)
        
        tag_sequences = []
        for i, sentence in enumerate(test_file):
            temp_list = []
            for word in sentence:
                temp_list.append(word[0])
            
            tag_sequences.append([(temp_list[j], y_predict[i][j]) for j in range(len(temp_list))])
        self.test_file_out(tag_sequences, file_out)
    
    def train_crf(self, x_training, y_training):
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        return crf.fit(x_training, y_training)
    
    def select_feature(self, word, tag, feature_conf, conf_switch):
        feature_dict = {
            'bias': 2.0,
            conf_switch + '_word.lower()': word.lower(),  
            conf_switch + '_word[-3]': word[-3:],  
            conf_switch + '_word[-2]': word[-2:],  
            conf_switch + '_word.isupper()': word.isupper(),  
            conf_switch + '_word.isdigit()': word.isdigit(),  
            conf_switch + '_word.islower()': word.islower(),
            conf_switch + '_tag': tag 
        }

        return {i: feature_dict.get(i) for i in feature_conf[conf_switch] if i in feature_dict.keys()}

    #Extract the features from the given word in the sentence
    def word_to_features(self, line, idx, feature_conf):

        word, tag = line[idx]
        feats = self.select_feature(word, tag, feature_conf, "current")
        if idx > 0:
           prev_word, prev_tag = line[idx-1]
           feats.update(self.select_feature(prev_word, prev_tag, feature_conf, "previous"))
        else:
            feats["BOS"] = True
        
        if idx < len(line) - 1:
            next_word, next_tag = line[idx+1]
            feats.update(self.select_feature(next_word, next_tag, feature_conf, "next"))
        else:
            feats["EOS"] = True
        return feats

    #Extract the features from the given word in the sentence, in a test
    def word_to_features_test(self, line, idx, feature_conf):

        word = line[idx]
        feats = self.select_feature(word, None, feature_conf, "current")
        if idx > 0:
           prev_word = line[idx-1]
           feats.update(self.select_feature(prev_word, None, feature_conf, "previous"))
        else:
            feats["BOS"] = True
        
        if idx < (len(line) - 1):
            next_word = line[idx+1]
            feats.update(self.select_feature(next_word, None, feature_conf, "next"))
        else:
            feats["EOS"] = True

        return feats

    def sent_features(self, line, feature_conf):
        return [self.word_to_features(line, k, feature_conf) for k in range(len(line))]
    
    def sent_features_test(self, line, feature_conf):
        return [self.word_to_features_test(line, k, feature_conf) for k in range(len(line))]

    def sent_labels(self, line):
        return [tag for _, tag in line]
    
    def test_file_out(self, data, out_file):
        with open(out_file, "w",  encoding="utf-8") as f:

            for line in data:
                for word, tag in line:

                    f.write(word + ' ' + tag + '\n')

                f.write("\n")

if __name__ == "__main__":

    train_data = tokenize(train_dir)
    crf = CRF()
    crf.fit(train_data)
    crf.predict(tokenize(test_dir), out_dir)