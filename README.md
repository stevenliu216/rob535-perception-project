# rob535-perception-project
Fall 2019 Final project for ROB535: Self Driving Cars

## Kaggle link
[https://www.kaggle.com/c/rob535-fall-2019-task-1-image-classification](link)

## Task 1
This is the first task for Fall 2019 ROB 535 - Self-driving Cars: Perception and Control. In this task, you are asked to classify a vehicle in a snapshot.

## Instructions
We rearranged the provided dataset (`utils/rearrange_data.py`) into a different format to better suit our needs in Pytorch. It was just simpler to do than implementing a custom dataset object. Run the following script to download and unzip our version of the dataset and our pretrained model.
```bash
$ ./setup_data.sh
```

Model training was performed in Colab Notebook which provides GPU resources. Take a look at 'notebooks/transfer_learning_.ipynb` for details. To evaluate our model, run the following command which will output a file called `predictions.csv` with the results:
```
python eval.py
```
