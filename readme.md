Disaster Response Classifier
==============================

Purpose of this project
-----------------------

This is a supervised learning classification project. 
Using pipelines (ETL and ML pipelines), a multi-output Random Forest Classifier was trained on the data after applying NLP techniques. 
A web app is available as an interface to use the ML model to predict the class of an inputted message and provide visualizations on the training data.
The focus of this project is the classification of messages sent during of after a disaster to the appropriate category (ex: water, shelter, fire, hospital...). The goal behind is to be able to assist forwarding the message to the appropriate recipient and therefore ensure a more efficient assistance of people concerned by the disaster.

In this repository, you will find:

- Two Jupyter notebooks that go through the process of defining the pipelines (ETL and ML pipelines)
- A folder containing the script to execute the ETL pipeline
- A folder containing the code related to training and saving the model
- A folder containing the files necessary to display the web app
- A folder containing the data (csv inputs and database)
- This readme file


Requirements
------------

The code is written in Python 3.7.

There are several dependencies for this package. The versions provided here are those used for testing.

- argparse 1.1
- json 2.0.9
- nltk 3.4.4
- numpy 1.16.4
- pandas 0.25.0
- pickle 4.0
- os 
- re 2.2.1
- sklearn 0.21.2
- sqlachemy 1.3



How to use the code
-------------------

There are two approaches on how to use this code:
- use the model as it is to predic the class of one or more messages
- train a new model with new data

In the case of the first approach, please take a look at the next section regarding the web app.
As for the second approach, there are several steps:
1. Execute the script process_data.py, passing as arguments:
	- the path to the disaster_messages.csv file
	- the path to the disaster_categories.csv file,
	- the path or name of the database where to store the data once processed 
2. Update the conf.json file to set the parameters of the train_classifier.py script
3. Execute the script train_classifier.py, passing as arguments:
	- the path or name of the database where to store the data once processed 
	- the path or name of the pickle file of the saved ML model


General comments about the process_data.py script:
- the loading of the data takes into account the fact that other formats can be provided for the messages and categories files (for example json, xml or tables of a db)
- the transformed and merged data is saved in a table named 'MessagesWithCategory' by default, which can be changed directly in the main() function of the code of the script

General comments about the train_classifier.py:
- It is structured as to allow flexibility in the way the model should be trained - parameters are set in a conf.json file allowing the user to fine tune which hyperparameters to use for CountVectorizer and RandomForestClassifier, as well as whether to use more advanced NLP transformers and grid_search
- In order to avoid needing to always train a model from scratch, if the pickle file provided as an argument exist (i.e a model was already saved), the model saved is used as a base for **further** training - the hyperparameters set in the conf.json file should therefore be chosen in consequence!


Web app
-------



Comments on the data
---------------------

Classes unbalanced
-------------------


Data cleaning
--------------


Data modeling
--------------

Go into more detail about the dataset and your data cleaning and modeling process in your README file, add screenshots of your web app and model results.

This dataset is imbalanced (ie some labels like water have few examples). In your README, discuss how this imbalance, how that affects training the model, and your thoughts about emphasizing precision or recall for the various categories.


Further work and improvements
-----------------------------

Some improvements that could be performed on the code of this project:

- use a conf.json file for the process_data.py script as well
- find a way to run GridSearch with n_jobs=-1 (and not only the classifier RandomForest)
- fine tune better the estimators hyperparameters as to keep the same level of accuracy but reduce process time
- one function (word_count_per_cat), used when setting full_txt_process to true, takes a long time to run - could be optimised


Sources, acknowlegments and related content
-------------------------------------------

The dataset used in this project was provided by Figure Eight.
This project is part of the UDACITY Data Scientist Nanodegree.

