# rob535-perception-project
Fall 2019 Final project for ROB535: Self Driving Cars. This is the perception part.

## Setup
Setup python3 virtual environment and install dependencies: 
> Caution: Only tested on MAC OSX.
> Caution: To exit the virtual environment, type `deactivate`
```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/c/rob535-fall-2019-task-1-image-classification/data). Alternatively, set up API token and use below command:
```bash
$ kaggle competitions download -c rob535-fall-2019-task-1-image-classification
```

Unzip the downloaded data into the `data/` directory as follows:
```
data
└── rob535-fall-2019-task-1-image-classification
```

We pretrained a model based on ResNeXt-50 and uploaded to google drive [here](https://drive.google.com/file/d/1-63b76I54aiyGQl5ZR_7Qkjm3RT-HpfN/view?usp=sharing). Before running the test code, download this model and place it in the `models/` directory:
```
models
└── team10_trained_resnext_epoch5.pt
```

## Task 1
The following command will use the pretrained model to evaluate the data. It will output a new file called `predictions.csv`. Warning: If on CPU, this will take a few minutes (~6 minutes):
```bash
$ python task1.py
```

To submit to Kaggle for evaluation:
```bash
$ kaggle competitions submit -c rob535-fall-2019-task-1-image-classification -f predictions.csv -m "new submission"
```

## Task 2
Optional for 20 bonus points. We did not try it yet.

## Additional information
The model is trained on a Tesla K80 GPU provided by Google Colab. Refer to `colab/transfer_learning.ipynb` for details. The final trained model is stored in [Google Drive](https://drive.google.com/file/d/1-63b76I54aiyGQl5ZR_7Qkjm3RT-HpfN/view?usp=sharing).

### Kaggle Competition links
[Task1](https://www.kaggle.com/c/rob535-fall-2019-task-1-image-classification)
[Task2](https://www.kaggle.com/c/rob535-fall-2019-task-2-vehicle-localization)