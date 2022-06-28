# Cats vs Dogs Example
This is an exemplary project on how to solve an image classification problem but in a proper manner.

The project includes following features:
* Batch data preprocessing based on Python data generators
* Use of Python data generators to read and preprocess data
* Employment of tf.Data.Dataset API for providing data for model training
* Use of Jupyter Notebooks for both data exploration and model evaluation (incl. class activation maps)
* DVC for pipeline designing and execution, see dvc.yaml
* Parameter definition in an external file params.yaml
* Storage of metrics in an external file metrics.json

Thus, this project after some adaptation is supposed to be a template for simple ML problems in the field of CV.

## Data
The project expects raw data in a certain format. Each instance should consist of an image and a json file with metadata (incl. label).
Currently, a publicly available dogs vs cats dataset is used (copyrights: https://www.microsoft.com/en-us/download/details.aspx?id=54765).
To download already properly formatted data please use the following link: https://cloudstore.zih.tu-dresden.de/index.php/s/8bMk2LKdaPDYHfA

The downloaded files need to be put in folder 'data' (this folder must be created by the user).

## DVC pipeline
To be able to run the pipeline specified in dvc.yaml you need first install DVC. After that you can run the entire pipeline executing the command:
```bash
dvc repro
```
DVC expects the user to have the project folder as a working directory.
Therefore, if you need to run single scripts in Python, please change the working directory of the interpreter to be the project directory.