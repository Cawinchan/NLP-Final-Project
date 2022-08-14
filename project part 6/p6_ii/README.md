# Project Part 6

## Part 6ii


All the required data files(including test.p6.model.out) are under 'data', with 'laptop14' containing active training files and 'test20' containing active test files. The rest of the files are outside and labelled accordingly. 

All the trained BERT models are under 'bert-tfm-laptop14-finetune', done in 15 checkpoints. 'bert-tfm-laptop14-finetune' is too large and hence we removed it and uploaded it to [here](https://drive.google.com/drive/folders/13elhJ1HCMYsoq3-ssCNvsRWGkVSHtR1r?usp=sharing)

## Installation 

```bash 
    pip install -r requirements.txt
```

## How to use

To prepare the dataset provided by the course for the model, run the  'converter.py' file after typing in the relevant paths and variables as shown:

``` python
    train_dir = Path(__file__).resolve().parent/"data"/"train"
    devin_dir = Path(__file__).resolve().parent/"data"/"dev.in"
    test_dir = Path(__file__).resolve().parent/"data"/"test.in"
    path_out = Path(__file__).resolve().parent/"data"/"convert_out_dev.txt"

    #use test_converter for datasets without labels and converter for datasets with labels
    if __name__ == "__main__":

    data_in = tokenize(test_dir)
    # converter(tokenize(test_dir) , path_out)
    test_converter(data_in, path_out)
```

To train the model on any dataset, place data files into directory ./data/[YOUR_DATASET_NAME] (In our case it is in ./data/laptop14), making sure to rename data files so that it can be directly adapted to this project, like ‘train.txt’, ‘test.txt’, etc.  Set TASK_NAME in train.sh as [YOUR_DATASET_NAME]. Train the model with the command: 
``` bash
    sh train.sh
```

To predict unlabelled data using the model trained, use ‘converter.py’ test_converter function to convert data provided by the course to a suitable one for the model. Place data file in the directory ./data/[YOUR_EVALUATION_DATASET_NAME](In our case it is under ./data/laptop14 ). Set TASK_NAME in work.sh as [YOUR_EVALUATION_DATASET_NAME].  Set ABSA_HOME in work.sh as [DIRECTORY_OF_MODEL]. (In our case it is ./bert-tfm-laptop14-finetune). 
In work.py, set the out_path accordingly to where you want the predicted file to be saved:
``` python
    out_path = Path(__file__).resolve().parent/'data'/'test20'/'test.p6.model.out'
```

Predict the model with the command: 
``` bash
    sh work.sh
```

Major credits to the research paper ‘Exploiting BERT for End-to-End Aspect-based Sentiment Analysis’: https://arxiv.org/pdf/1910.00883.pdf 
