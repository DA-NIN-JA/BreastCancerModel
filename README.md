#Breast Cancer Prediction
##Overview
This project aims to predict whether a breast cancer tumor is malignant or benign using machine learning. The model is built with the scikit-learn library in Python and deployed as a web application using Flask.

##Features
Clump Thickness: Describes the thickness of the clump.
Uniformity of Cell Size: Measures the uniformity of cell size.
Uniformity of Cell Shape: Measures the uniformity of cell shape.
Marginal Adhesion: Describes how cells are connected to each other.
Single Epithelial Cell Size: Measures the size of epithelial cells.
Bare Nuclei: Describes whether the nucleus is bare.
Bland Chromatin: Describes the chromatin's appearance.
Normal Nucleoli: Describes the appearance of normal nucleoli.
Mitoses: Measures the number of mitoses.
##Model
The model is based on logistic regression, a popular algorithm for binary classification tasks. It is trained on a dataset with various features related to breast cancer tumors.

##Web Application
The machine learning model is integrated into a web application using Flask. Users can input tumor characteristics, and the application predicts whether the tumor is malignant or benign.
