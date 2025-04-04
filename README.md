**README File to explain the functioning of the AI Model Trainer:**

Although I provided the code, the application is fully functional and deployed in the Streamlit Community Cloud, if you prefer you can use the application using this link: https://app-mltrainer-victorperez.streamlit.app/

Interactive Machine Learning Model Trainer: This application allows users to interactively train and evaluate basic machine learning models (both Regression and Classification) on sample datasets or their own uploaded data.

Data Source Selection (you can choose from 2 options):
- Choose from sample Seaborn datasets (Iris, Penguins, Tips).
- Upload your own data in CSV format.

Interactive Configuration:
- Select the target variable (y)
- Choose features (X) from available quantitative and qualitative columns.
- Select the machine learning model (Linear Regression, Random Forest Regressor/Classifier based on task type)
- Configure training parameters (Test Set Size, Random State).
- Adjust hyperparameters for each model (for example: select the `n_estimators`, `max_depth` for Random Forest).

Controlled Execution:
- Model training and evaluation only occur when the "Train & Evaluate" button is pressed, thanks to the use of `st.form`.

Performance Evaluation:
- Displays key metrics based on the task (RMSE/R² for Regression, Accuracy/F1/Precision/Recall/AUC for Classification).

Visualizations:
- Regression: Residual Distribution Plot, Feature Importance Plot.
- Classification: Confusion Matrix, ROC Curve (Micro-Average for Multiclass), Feature Importance Plot.
- Session History: Tracks configuration and results of successful runs within the current browser session in the sidebar.
- Model Download: Allows downloading the trained pipeline (including preprocessor and model) as a `.pkl` file using `joblib`.

IMPORTANT: I chose the Micro-Average ROC Curve because it gives a single, combined measure of the performance across all classes. This measure treats every sample the same, regardless of its class, which makes it very useful when you want a global summary of how well the model distinguishes between any correct class and any incorrect class.

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
