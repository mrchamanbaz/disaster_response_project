# Disaster Response Pipeline Project


## Description of the Project:
This Project is a part of Data Science Nanodegree Program by Udacity. The goal of the project is to develop a pipeline that predicts the category of incoming messages related to disaster response. The training dataset is a prelabelled set of messages provided in two files `disaster_messages.csv` (for the messages) and `disaster_categories.csv` (for the labels). A data dashboard is also developed so that the user can input new messages and finds out the corresponding categories of the message. 

## Project Structure 

The Project is divided into the following Sections:

1. Data Processing: Data processing is performed in the file `ETL Pipeline Preparation.ipynb`. The same code is also included in `process_data.py`. This file reads the datasets, cleans and merge them together and saves the result in a sql database. 
2. Machine Learning Pipeline: Designing and learning the classifier is performed in `ML Pipeline Preparation.ipynb` as well as the script `train_classifier.py`. In the jupyter notebook, we use a grid search as well as a neural network classifier (using tensorflow library) to train a model which is able to classify text messages in 36 categories. The best model turns out to be the tuned [K Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) model returned by a gridsearch over extensive simulation. The parameters leading to the best model are used in `train_classifier.py` to actually train and save the classifier as a pickle file.
3. Web Application using Flask: This part is done in `run.py`. It creates a number of data explanatory visualizations as well as a dashboard to classify new messages. 

## Data:

The data in this project comes from Figure Eight - Multilingual Disaster Response Messages. Data is presented in two `.csv` files: 
1. `disaster_messages.csv` which includes messages
2. `disaster_categories.csv` which includes labels for each message.


## Required packages:

This project requires Python 3.x and the following Python libraries:

* Machine Learning Libraries: NumPy, Pandas, Sciki-Learn, tensorflow
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

## How to Run the Project:

1. Clone the repository

```
git clone git@github.com:mrchamanbaz/disaster_response_project.git
```

2. Run the following commands to generate the database and model.

    - To perform data processing run the following code in data's directory

        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To design the classifier, run the following code in models' directory

        `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

3. Run the following command in the app's directory to run the web app.

    `python run.py`

4. Go to the link specified by the flask app. For example if the code returns the following: 
```
Debugger PIN: 418-088-814

Running on all addresses.

Running on http://10.48.24.219:3001/ (Press CTRL+C to quit)
```
You should open http://10.48.24.219:3001/ in the browser to see the web app.

## Appearance:
The web app homepage looks like this:


![Screenshot_1](https://github.com/mrchamanbaz/disaster_response_project/blob/b61b4c282dd8e91c477eddd535cdc2e8d481a949/screenshots/home_1.jpg?raw=true "Title")

In the homepage, you can see three graphs. The first graph shows the distribution of the message genres. The second and third graphs shown below present the histogram of 36 categories in the training dataset and 20 most frequent words after removing common words like 'the', 'of', 'a', etc.

![Screenshot_1](https://github.com/mrchamanbaz/disaster_response_project/blob/b61b4c282dd8e91c477eddd535cdc2e8d481a949/screenshots/home_2.jpg?raw=true "Title")

On top of the homepage one can insert a new message and click 'classify message'. Doing this would take the user to the next page running the classification algorithm on the new message and returning back the predicted classes. We note that the problem is a multioutput classification problem then multiple categories (classes) can be predicted for a message. 

![Screenshot_1](https://github.com/mrchamanbaz/disaster_response_project/blob/b61b4c282dd8e91c477eddd535cdc2e8d481a949/screenshots/message_category.jpg?raw=true "Title")

## Licensing, Authors, and Acknowledgements

Please feel free to use the code here as you would like. 