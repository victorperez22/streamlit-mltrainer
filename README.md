Interactive Machine Learning Model Trainer

This Streamlit application allows users to interactively train and evaluate basic machine learning models (Regression and Classification) on sample datasets or their own uploaded data.

This project was created as part of the Data Visualization assignment.

Data Source Selection:
- Choose from sample Seaborn datasets (Iris, Penguins, Tips).
- Upload your own data in CSV format.

Interactive Configuration:
- Select the target variable (y).
- Choose features (X) from available quantitative and qualitative columns.
- Select the machine learning model (Linear Regression, Random Forest Regressor/Classifier based on task type).
- Configure training parameters (Test Set Size, Random State).
- Adjust model-specific hyperparameters (for example: `n_estimators`, `max_depth` for Random Forest).

Controlled Execution: 
- Model training and evaluation only occur when the "Train & Evaluate" button is pressed, thanks to the use of `st.form`.

Performance Evaluation:**
- Displays key metrics based on the task (RMSE/R² for Regression, Accuracy/F1/Precision/Recall/AUC for Classification).

Visualizations:
- Regression: Residual Distribution Plot, Feature Importance Plot.
- Classification: Confusion Matrix, ROC Curve (Micro-Average for Multiclass), Feature Importance Plot.
- Session History: Tracks configuration and results of successful runs within the current browser session in the sidebar.
- Model Download: Allows downloading the trained pipeline (including preprocessor and model) as a `.pkl` file using `joblib`.

How to Use the Deployed App

1.  Select Data: Use the sidebar to choose a sample dataset or upload your own CSV file.
2.  Preview Data: Examine the first few rows, shape, and basic info of the loaded data. Optionally expand "Raw Data Details" for more info.
3.  Configure:
- Go to the "Configuration" section in the main area.
- Select your Target Variable (y).
- Select the Features (X) you want to use (Quantitative and Qualitative).
- Verify the automatically Detected Task Type.
- Choose the Model type.
- Adjust Training Parameters (Test Size, Random State).
 - Tune Model Hyperparameters.
4.  Train & Evaluate: Click the "🚀 Train & Evaluate" button at the bottom of the form.
5.  View Results: Examine the performance metrics and visualizations generated for the test set.
6.  Check History: Look at the sidebar under "📜 Training History" to see summaries of previous successful runs in this session.
7.  Download Model: If desired, click the "Download Trained Model Pipeline (.pkl)" button that appears after successful training.

Technologies used (these are the libraries used in the project)
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Numpy

The application is fully functional and deployed in the Streamlit Community Cloud