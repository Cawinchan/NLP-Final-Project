The code for question 6 part i is under the directory of ‘p6_i’. 
All the required data files(including test.p6.CRF.out)  are under ‘dataset’, and the main file to run is ‘main.py’.
Dependencies required to run this part are in ‘requirements.txt’.

Simply run the python file to train the data using the provided ‘train’ file, and predict the provided ‘test.in’ file. The file with the predicted data will be created to out_dir after running main.py. Change the paths and variables accordingly to what file you want to train and predict, as well as the output file path. You may do the change in lines under main.py:

train_dir = Path(__file__).resolve().parent/"dataset"/"train"
devin_dir = Path(__file__).resolve().parent/"dataset"/"dev.in"
test_dir = Path(__file__).resolve().parent/"dataset"/"test.in"
out_dir = Path(__file__).resolve().parent/'dataset'/"test.p6.CRF.out"
 

if __name__ == "__main__":
    train_data = tokenize(train_dir)
    crf = CRF()
    crf.fit(train_data)
    crf.predict(tokenize(test_dir), out_dir)