# Funeral Insurance Uptake Model

This repository contains a developed model for predicting funeral insurance uptake. The model is accompanied by a Jupyter Notebook and a Flask web application that allows users to make predictions based on the saved model.

## Notebook

The `funeral-insurance-uptake-logistic-regression-model.ipynb` notebook provides an in-depth analysis of the funeral insurance data and demonstrates the process of developing the logistic regression model. It covers data preprocessing, feature engineering, model training, evaluation, and saving the trained model.

## Flask App

The Flask web application allows users to interact with the trained model and make predictions. The app consists of a single page where users can input relevant information, such as age, income, occupation, and family structure, and receive a prediction on the likelihood of funeral insurance uptake.

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- Flask
- scikit-learn
- pandas
- numpy

### Usage

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask app using `python app.py`.
4. Access the app in your browser at `http://localhost:5000`.
5. Enter the required information in the input fields.
6. Click the "Predict" button to get the prediction result.

Feel free to modify the Flask app as needed, such as adding additional features or improving the user interface.

## Acknowledgements

This project is based on the research topic of developing a logistic regression model to predict funeral insurance uptake for Ecosure Zimbabwe. We would like to thank Ecosure Zimbabwe for providing the necessary data and resources for this project.

Please note that this model is for demonstration purposes only and should be further refined and validated before being deployed in a production environment.

## License

[MIT License](LICENSE)

Feel free to customize this template and add more sections based on the specific details and requirements of your developed model.