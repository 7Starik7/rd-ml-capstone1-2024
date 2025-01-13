# Loan traffic sign recognition service

## Overview

This project focuses on developing a machine learning model 
to classify images of traffic signs into their respective categories. 
Traffic sign recognition is a critical component 
of advanced driver-assistance systems. An accurate model will ensure 
safer navigation, streamline vehicle decision-making, and reduce the risk 
of accidents by correctly interpreting road signs.

## Problem Statement

The ability to accurately recognize and classify traffic signs 
is vital for ensuring road safety and effective navigation 
in driver-assistance systems. This project aims to build a robust model 
that automates the recognition and classification of traffic signs from images, 
leveraging a labeled dataset of German road signs to achieve high accuracy 
and reliability in real-world scenarios.

**The next dataset will be used for traffic sign recognition:**

https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset/code

##### 1. How To get the data
Execute **_Data retreival_** block
from [notebook.ipynb](notebooks%2Fnotebook.ipynb)
to extract the data ([loan_data.csv](datasets%2Floan_data.csv) will be extracted).

##### 2. Perform Exploratory data analysis (EDA)
Execute **_Exploratory data analysis (EDA)_** block
from [notebook.ipynb](notebooks%2Fnotebook.ipynb) to perform EDA.

##### 3. Extract the images
Execute **_Images extraction_** block
from [notebook.ipynb](notebooks%2Fnotebook.ipynb).

##### 4. Define CNN model
Execute **_CNN model_** block
from [notebook.ipynb](notebooks%2Fnotebook.ipynb) to define the model.

##### 5. Create data generator
Execute **_Create the data generator, load data_** block
from [notebook.ipynb](notebooks%2Fnotebook.ipynb).

##### 6. # Train the model (default)
Execute **_Train the model_** and **_Evaluate the model_** blocks
from [notebook.ipynb](notebooks%2Fnotebook.ipynb).

##### 7. # Tune the model
Execute **_Learning rate adjustment_** and **_Drop rate adjustment_** blocks
from [notebook.ipynb](notebooks%2Fnotebook.ipynb).

##### 8. # Save the model
Execute **_Save best model_** blocks
from [notebook.ipynb](notebooks%2Fnotebook.ipynb).