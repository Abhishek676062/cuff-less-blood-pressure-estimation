# Cuff-less Blood Pressure Estimation using Machine Learning

This project aims to build a predictive model for a cuff-less blood pressure estimation device that uses machine learning to predict blood pressure based on PPG and ECG data. The model will take PPG and ECG data as input and predict the corresponding systolic and diastolic blood pressure readings.

# Dataset
We can not add dataset because it size is more than 5GB.

The dataset for this project is available on Kaggle as a ZIP file named cuff-less blood pressure estimation.zip. 

The dataset contains a train folder and a test folder. 

The train folder contains training data in the form of PPG and ECG signals, along with their corresponding systolic and diastolic blood pressure readings. 

The test folder contains test data in the same format as the training data.

# Install cuff-less-blood-pressure-estimation

```bash
  mkdir cuff-less-blood-pressure-estimation
  git clone https://github.com/Abhishek676062/cuff-less-blood-pressure-estimation.git
  cd cuff-less-blood-pressure-estimation
  py -m venv .venv
  pip install -r requirements.txt
```
# Requirements

To build and run the model, you will need the following:

Python 3.x

NumPy

Pandas

Matplotlib

Scikit-learn

Keras

TensorFlow

# Building the Model

Load the dataset from the train folder into memory using Pandas.

Preprocess the dataset as necessary to prepare it for training and testing the models. 

This may include normalizing the data, splitting it into training and validation sets, and converting it to a suitable format for the model.

Build a model using either LSTM, ANN, or CNN to predict systolic and diastolic blood pressure based on PPG and ECG data. 

The choice of model will depend on the characteristics of the data and the problem being solved.

Train the model using the preprocessed dataset. This may involve adjusting hyperparameters and using techniques such as early stopping to avoid overfitting.

Evaluate the performance of the model using appropriate metrics such as mean squared error (MSE) and root mean squared error (RMSE).

Test the model on the data in the test folder.

Submit the predicted systolic and diastolic blood pressure readings for the test data in the format specified in the sample_submission.csv file provided in the dataset.

# Evaluation Criteria

The model will be evaluated based on the following criteria:

Correctness of the implementation

Clarity and readability of the code

Performance of the model on the test data

Quality of the predicted blood pressure readings

# Conclusion

This project presents an approach to build a predictive model for a cuff-less blood pressure estimation device using machine learning. By leveraging PPG and ECG data, the model can predict systolic and diastolic blood pressure readings with high accuracy. The model can potentially be used in real-world applications to improve the monitoring and management of blood pressure.
