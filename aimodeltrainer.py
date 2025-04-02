import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support, make_scorer,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
import io
import joblib
import traceback
from itertools import cycle
from datetime import datetime

st.set_page_config(layout="wide", page_title="ML Model Trainer")
st.title("🤖 Interactive Machine Learning Model Trainer")
st.write("Configure and run ML models. Results are stored in the sidebar history for this session.")

#Used to initilizate the session state for each user.
default_session_state = {
    "model_bytes": None, "label_classes": None, "last_model_type": None,
    "prev_data_source": None, "prev_uploaded_file_id": None,
    "training_history": [] #list to store the different AI models trained
}
for key, default_value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = default_value

#Helper Functions
@st.cache_data
def load_data(source, uploaded_file_content=None, filename=None):
    try:
        if source == 'Upload':
            if uploaded_file_content: df = pd.read_csv(io.BytesIO(uploaded_file_content)); st.success(f"Loaded: {filename}"); return df, None
            else: return None, "Please upload a CSV file."
        else:
            df = sns.load_dataset(source);
            for col in df.select_dtypes(include=['category']).columns: df[col] = df[col].astype('object')
            return df, None
    except Exception as e: st.error(f"Error loading '{filename or source}': {e}"); return None, f"Error loading dataset: {e}"

@st.cache_data
def get_column_types(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist(); datetime_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()
    quantitative_cols = [str(col) for col in numeric_cols if col not in datetime_cols]; qualitative_cols = [str(col) for col in df.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist()]
    datetime_cols = [str(col) for col in datetime_cols]
    return quantitative_cols, qualitative_cols, datetime_cols

#Plotting functions
def plot_confusion_matrix(y_true, y_pred, classes):
    try:
        unique_labels_in_data = np.unique(np.concatenate((y_true, y_pred))); labels_for_cm = np.arange(len(classes)) if np.max(unique_labels_in_data) < len(classes) else np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm); fig, ax = plt.subplots(figsize=(max(4, len(classes)*0.6), max(3, len(classes)*0.5)))
        tick_labels = classes if len(labels_for_cm) == len(classes) else labels_for_cm
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion Matrix'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout(); return fig
    except Exception as e: st.warning(f"CM plot error: {e}"); return None

def plot_roc_curve(y_true, y_pred_proba, classes):
    if y_pred_proba is None or y_true is None: st.warning("ROC needs labels & probabilities."); return None
    n_classes = len(classes); fig, ax = plt.subplots(figsize=(7, 5))
    if n_classes <= 2: # Binary case
        y_true_binary = label_binarize(y_true, classes=np.unique(y_true)); pos_label_idx = 1 if y_true_binary.shape[1] > 1 else 0
        if y_true_binary.shape[1] == 1: ax.text(0.5, 0.5, 'ROC N/A\n(1 class only)', ha='center', va='center', transform=ax.transAxes)
        else:
            if y_pred_proba.shape[1] == 1: fpr, tpr, _ = roc_curve(y_true_binary[:, pos_label_idx], y_pred_proba[:, 0]); roc_auc = auc(fpr, tpr); label_text = f'ROC (area = {roc_auc:0.2f})'
            elif y_pred_proba.shape[1] >= 2: fpr, tpr, _ = roc_curve(y_true_binary[:, pos_label_idx], y_pred_proba[:, pos_label_idx]); roc_auc = auc(fpr, tpr); pos_class_name = classes[pos_label_idx] if pos_label_idx < len(classes) else f"Class {pos_label_idx}"; label_text = f'ROC of {pos_class_name} (area = {roc_auc:0.2f})'
            else: st.error("Inconsistent probability shape."); return None
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=label_text); ax.set_title('ROC Curve')
    else:
        try:
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes)); fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel()); roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'Micro-average ROC (area = {roc_auc:0.2f})', color='deeppink', linestyle=':', linewidth=4); ax.set_title('Micro-Average ROC')
        except Exception as mc_roc_e: st.warning(f"Micro-ROC plot error: {mc_roc_e}"); ax.text(0.5, 0.5, 'ROC plot error', ha='center', va='center', transform=ax.transAxes)
    ax.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--'); ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.legend(loc="lower right", fontsize='small'); plt.tight_layout(); return fig
# I chose the micro-average ROC curve because it provides a single combined measure of performance across all classes, which is better for the main objective of this assignment.

def plot_residuals(y_true, y_pred):
    try:
        residuals = np.array(y_true) - np.array(y_pred); fig, ax = plt.subplots(figsize=(8, 4)); sns.histplot(residuals, kde=True, ax=ax)
        ax.set_xlabel('Residuals (Actual - Predicted)'); ax.set_ylabel('Frequency'); ax.set_title('Residual Distribution'); plt.tight_layout(); return fig
    except Exception as e: st.warning(f"Residual plot error: {e}"); return None

def plot_feature_importance(model, feature_names, model_type):
    importances = None
    try:
        if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
        elif hasattr(model, 'coef_'): importances = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else: return None
        if not isinstance(feature_names, list): feature_names = list(feature_names)
        if importances is None or len(importances) != len(feature_names): st.warning(f"FI plot error: Mismatch ({len(importances) if importances is not None else 'None'}) vs ({len(feature_names)})."); return None
        indices = np.argsort(importances)[::-1]; max_features_to_show = 30
        if len(importances) > max_features_to_show: indices = indices[:max_features_to_show]; st.caption(f"Top {max_features_to_show} features shown.")
        fig, ax = plt.subplots(figsize=(10, max(6, len(indices) * 0.3))); ax.barh(range(len(indices)), importances[indices][::-1], align='center')
        ax.set_yticks(range(len(indices))); ax.set_yticklabels(np.array(feature_names)[indices][::-1])
        ax.set_title('Feature Importance'); ax.set_xlabel('Importance Score' if hasattr(model, 'feature_importances_') else 'Absolute Coefficient'); plt.tight_layout(); return fig
    except Exception as e: st.warning(f"FI plot error: {e}"); return None

#Sidebar configurations
st.sidebar.header("1. Data Source")
available_datasets = ['iris', 'penguins', 'tips']
data_source_options = ['Upload'] + available_datasets
selected_source = st.sidebar.selectbox("Select Data Source", data_source_options, key="data_source_selector")
uploaded_file = None; uploaded_file_content = None; uploaded_file_id = None
if selected_source == 'Upload':
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="file_uploader")
    if uploaded_file: uploaded_file_content = uploaded_file.getvalue(); uploaded_file_id = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"

#History diplayed in the sidebar
st.sidebar.header("📜 Training History")
if st.session_state.training_history:
    #Clear history button
    if st.sidebar.button("Clear Run History"):
        st.session_state.training_history = []
        st.rerun()

    for i, run in enumerate(reversed(st.session_state.training_history)):
        run_index = len(st.session_state.training_history) - 1 - i
        timestamp = run.get('timestamp', 'N/A')
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = str(timestamp)

        expander_title = f"Run {run_index + 1}: {run['config'].get('model_type', 'N/A')} on {run['config'].get('dataset', 'N/A')} ({timestamp_str})"

        with st.sidebar.expander(expander_title):
            st.markdown("**Configuration:**")
            st.json(run['config'], expanded=False)
            st.markdown("**Results:**")
            #Display the metrics
            if run.get('results'):
                for metric, value in run['results'].items():
                    try:
                        st.metric(label=metric.replace('_', ' ').title(), value=f"{value:.4f}")
                    except (TypeError, ValueError):
                         st.metric(label=metric.replace('_', ' ').title(), value=str(value)) # Fallback for non-numeric
            else:
                st.caption("No results recorded.")
else:
    st.sidebar.caption("No training runs recorded yet.")
st.sidebar.markdown("---")

#Loading the Data
df = None; error_message = None
#Reset the model if the data changes
if st.session_state.prev_data_source != selected_source or st.session_state.prev_uploaded_file_id != uploaded_file_id:
    st.session_state.model_bytes = None; st.session_state.last_model_type = None; st.session_state.label_classes = None;
    st.session_state.prev_data_source = selected_source; st.session_state.prev_uploaded_file_id = uploaded_file_id
if selected_source == 'Upload':
    if uploaded_file_content: df, error_message = load_data(selected_source, uploaded_file_content, uploaded_file.name)
    elif not st.session_state.prev_uploaded_file_id: st.info("Awaiting CSV file upload...")
elif selected_source: df, error_message = load_data(selected_source)

#Main Area
if error_message: st.error(error_message)

if df is not None:
    st.header("📊 Data Preview")
    st.dataframe(df.head()); st.write(f"Shape: {df.shape}")
    with st.expander("Show Raw Data Details"):
        st.subheader("Data Info"); buffer = io.StringIO(); df.info(buf=buffer); s = buffer.getvalue(); st.text_area("DataFrame Info:", s, height=150, key="df_info_area")
        st.subheader("Numerical Stats"); st.dataframe(df.describe(include=np.number))
        st.subheader("Categorical Statistics"); categorical_cols = df.select_dtypes(include=['object', 'category', 'boolean']).columns
        if not categorical_cols.empty: st.dataframe(df.describe(include=['object', 'category', 'boolean']))
        else: st.write("No categorical, object, or boolean columns found.")

    st.header("⚙️ Configuration")
    #Configuration Form (to adjust and personalize parameters of the AI Model)
    with st.form("ml_config_form"):
        st.subheader("2. Feature Selection")
        quantitative_cols, qualitative_cols, datetime_cols = get_column_types(df); all_potential_columns = [str(c) for c in df.columns]
        if not all_potential_columns: st.error("No columns found."); st.stop()
        default_target_index = len(all_potential_columns) - 1; common_targets = ['target', 'class', 'species', 'survived', 'fare', 'price', 'tip', 'y', 'output', 'result']
        for i, col in enumerate(all_potential_columns):
             if col.lower() in common_targets: default_target_index = i; break
        target_variable = st.selectbox("Select Target (y)", all_potential_columns, index=default_target_index, key="target_selector")
        target_variable_str = str(target_variable)
        quantitative_features_options = [col for col in quantitative_cols if col != target_variable_str]; qualitative_features_options = [col for col in qualitative_cols if col != target_variable_str]
        st.info("Note: Datetime features excluded."); st.write("**Select Features (X):**")
        if quantitative_features_options: selected_quantitative_features = st.multiselect("Quantitative Features", quantitative_features_options, key="quant_feature_selector")
        else: st.caption("No quantitative features available."); selected_quantitative_features = []
        if qualitative_features_options: selected_qualitative_features = st.multiselect("Qualitative Features", qualitative_features_options, key="qual_feature_selector")
        else: st.caption("No qualitative features available."); selected_qualitative_features = []
        selected_features = selected_quantitative_features + selected_qualitative_features
        if not selected_features and (quantitative_features_options or qualitative_features_options): st.warning("Select at least one feature.")
        elif not quantitative_features_options and not qualitative_features_options: st.warning("No features available to select.");

        task_type = None
        if target_variable:
             try:
                  target_series = df[target_variable].dropna(); target_dtype = target_series.dtype; n_unique_target = target_series.nunique()
                  if pd.api.types.is_numeric_dtype(target_dtype) and not pd.api.types.is_datetime64_any_dtype(target_dtype):
                      if n_unique_target < 25 or pd.api.types.is_integer_dtype(target_dtype): task_type = "Classification"
                      else: task_type = "Regression"
                  elif pd.api.types.is_datetime64_any_dtype(target_dtype): task_type = "Regression"
                  else: task_type = "Classification"
                  if task_type: st.write(f"**Detected Task Type:** {task_type}")
             except Exception: pass

        st.subheader("3. Model & Training Parameters")
        if task_type == "Regression": model_options = ["Linear Regression", "Random Forest Regressor"]
        elif task_type == "Classification": model_options = ["Random Forest Classifier"]
        else: model_options = ["Linear Regression", "Random Forest Regressor", "Random Forest Classifier"]
        selected_model_type = st.selectbox("Select Model", model_options, key="model_selector")

        #Training options (test/train split)
        test_size = st.slider("Test Set Size (%)", 10, 50, 25, 5, key="test_size_slider") / 100.0
        random_state = st.number_input("Random State", value=42, step=1, key="random_state_input")

        #Hyperparameters the model needs to get defined beforehand
        st.markdown("**Model Hyperparameters:**")
        model_params_config = {} 
        if selected_model_type == "Linear Regression": model_params_config['fit_intercept'] = st.checkbox("Fit Intercept", True, key="lr_fit_intercept")
        elif selected_model_type in ["Random Forest Regressor", "Random Forest Classifier"]:
             model_params_config['n_estimators'] = st.slider("Num Trees (n_estimators)", 10, 500, 100, 10, key="rf_n_estimators")
             max_depth_val = st.slider("Max Depth (max_depth, 0=None)", 0, 50, 10, 1, key="rf_max_depth");
             model_params_config['max_depth'] = None if max_depth_val == 0 else max_depth_val
             model_params_config['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2, 1, key="rf_min_samples_split")
             model_params_config['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 20, 1, 1, key="rf_min_samples_leaf")
             if task_type == "Classification" and selected_model_type == "Random Forest Classifier": model_params_config['criterion'] = st.selectbox("Criterion", ["gini", "entropy"], 0, key="rf_criterion")

        submitted = st.form_submit_button("🚀 Train & Evaluate")
    #End of form

    #Action after submission
    if submitted:
        st.session_state.model_bytes = None; st.session_state.last_model_type = None; st.session_state.label_classes = None;

        if not quantitative_features_options and not qualitative_features_options: st.error("❌ No features available."); st.stop()
        if not selected_features: st.error("❌ Select features."); st.stop()
        if not target_variable: st.error("❌ Select target."); st.stop()
        if not selected_model_type: st.error("❌ Select model."); st.stop()
        if task_type is None: st.error("❌ Task type unclear."); st.stop()
        if task_type == "Regression" and "Classifier" in selected_model_type: st.error(f"❌ Invalid: {selected_model_type} for Regression."); st.stop()
        if task_type == "Classification" and "Regressor" in selected_model_type: st.error(f"❌ Invalid: {selected_model_type} for Classification."); st.stop()

        st.header("🚀 Training & Evaluation")
        try:

            run_config = {
                "dataset": selected_source if selected_source != 'Upload' else uploaded_file.name,
                "target_variable": target_variable,
                "selected_features": selected_features,
                "model_type": selected_model_type,
                "test_size": test_size,
                "random_state": random_state,
                "hyperparameters": model_params_config # Use the captured params
            }

            #Data preparation
            df_train = df.dropna(subset=[target_variable]).copy();
            if df_train.shape[0] < df.shape[0]: st.warning(f"Dropped {df.shape[0] - df_train.shape[0]} rows with missing target.")
            if df_train.empty or df_train.shape[0] < 10: st.error("❌ Insufficient data after NA drop."); st.stop()
            cols_to_keep = selected_features + [target_variable]
            df_processed = df_train[[col for col in cols_to_keep if col in df_train.columns]].copy()
            final_selected_quantitative = [f for f in selected_quantitative_features if f in df_processed.columns]; final_selected_qualitative = [f for f in selected_qualitative_features if f in df_processed.columns]
            final_selected_features = final_selected_quantitative + final_selected_qualitative
            if not final_selected_features: st.error("❌ No selected features remain after NA drop."); st.stop()
            X = df_processed[final_selected_features]; y_original = df_processed[target_variable]

            #Using pipeline for preprocessing
            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]); categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, final_selected_quantitative), ('cat', categorical_transformer, final_selected_qualitative)], remainder='drop', verbose_feature_names_out=False); preprocessor.set_output(transform="pandas")

            #Model definition and Target encoding
            le = None; y_to_use = y_original
            #Use  captured configuration hyperparameters to be able to define the model
            model_params_to_use = run_config['hyperparameters']
            if task_type == "Classification":
                le = LabelEncoder(); y_encoded = le.fit_transform(y_original); st.session_state['label_classes'] = le.classes_ ; y_to_use = y_encoded
                model = RandomForestClassifier(random_state=random_state, n_jobs=-1, **model_params_to_use)
            elif task_type == "Regression":
                if selected_model_type == "Linear Regression": model = LinearRegression(**model_params_to_use)
                elif selected_model_type == "Random Forest Regressor": model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params_to_use)
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

            #the final split between training and testing datasets
            st.subheader("🎓 Model Training")
            try: X_train, X_test, y_train, y_test_final = train_test_split(X, y_to_use, test_size=test_size, random_state=random_state, stratify=y_to_use if task_type=="Classification" and len(np.unique(y_to_use)) > 1 else None)
            except ValueError as split_error: st.warning(f"Stratified split failed ({split_error}), using non-stratified split."); X_train, X_test, y_train, y_test_final = train_test_split(X, y_to_use, test_size=test_size, random_state=random_state)
            with st.spinner(f"Training {selected_model_type}..."): pipeline.fit(X_train, y_train)
            st.success(f"✅ Model training complete!")

            #Evaluation on test set
            st.subheader("🧪 Test Set Evaluation")
            y_pred_test = pipeline.predict(X_test); y_pred_proba_test = None
            run_results = {} #dictionary created to store results
            if task_type == "Classification" and hasattr(pipeline, "predict_proba"):
                try: y_pred_proba_test = pipeline.predict_proba(X_test)
                except Exception as proba_e: st.warning(f"Could not get test set probabilities: {proba_e}")
            col1, col2 = st.columns(2)
            if task_type == "Regression":
                rmse_test = np.sqrt(mean_squared_error(y_test_final, y_pred_test)); r2_test = r2_score(y_test_final, y_pred_test)
                col1.metric("Test RMSE", f"{rmse_test:.4f}"); col2.metric("Test R²", f"{r2_test:.4f}")
                run_results = {"RMSE": rmse_test, "R2": r2_test}
            elif task_type == "Classification":
                 accuracy_test = accuracy_score(y_test_final, y_pred_test); precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test_final, y_pred_test, average='weighted', labels=np.unique(y_test_final), zero_division=0)
                 col1.metric("Test Accuracy", f"{accuracy_test:.4f}"); col2.metric("Test Weighted F1", f"{f1_test:.4f}"); st.markdown(f"**Test Precision (W):** {precision_test:.4f} | **Test Recall (W):** {recall_test:.4f}")
                 run_results = {"Accuracy": accuracy_test, "F1_Weighted": f1_test, "Precision_Weighted": precision_test, "Recall_Weighted": recall_test}
                 #Adding AUC if available to store results
                 if y_pred_proba_test is not None:
                      try:
                           n_classes_eval = len(st.session_state['label_classes'])
                           auc_kwargs = {'average': 'weighted', 'multi_class': 'ovr'} if n_classes_eval > 2 else {}
                           auc_score_test = roc_auc_score(y_test_final, y_pred_proba_test, **auc_kwargs)
                           run_results["AUC_Weighted"] = auc_score_test
                           st.markdown(f"**Test ROC AUC (Weighted):** {auc_score_test:.4f}")
                      except Exception as auc_err:
                           st.warning(f"Could not calculate test ROC AUC: {auc_err}")


            #Visualization of Test set
            st.subheader("📊 Test Set Visualizations")
            plot_col1, plot_col2 = st.columns(2); fig_res_cm = None
            if task_type == "Regression": fig_res_cm = plot_residuals(y_test_final, y_pred_test)
            elif task_type == "Classification": classes_cm = st.session_state.get('label_classes', np.unique(y_original)); fig_res_cm = plot_confusion_matrix(y_test_final, y_pred_test, classes_cm)
            if fig_res_cm: plot_col1.pyplot(fig_res_cm, clear_figure=True); plot_col1.caption("Residuals / CM (Test Set)")
            try: #The plot of feature importance
                 final_preprocessor = pipeline.named_steps['preprocessor']; num_features = final_selected_quantitative; ohe_features = []
                 cat_transformer_info = next((t for t in final_preprocessor.transformers_ if t[0] == 'cat'), None)
                 if cat_transformer_info:
                     cat_pipeline = cat_transformer_info[1]
                     if isinstance(cat_pipeline, Pipeline) and 'onehot' in cat_pipeline.named_steps:
                         onehot_encoder = cat_pipeline.named_steps['onehot']
                         if hasattr(onehot_encoder, 'get_feature_names_out') and hasattr(onehot_encoder, 'n_features_in_') and len(final_selected_qualitative) > 0:
                             try: ohe_features = list(onehot_encoder.get_feature_names_out(final_selected_qualitative))
                             except Exception: ohe_features = [f"cat_feat_{i}" for i in range(onehot_encoder.n_features_out_)] # Fallback names
                 processed_feature_names = num_features + ohe_features;
                 if not processed_feature_names: st.warning("No feature names for importance plot."); raise ValueError("Empty feature names")
                 final_model = pipeline.named_steps['model']
                 fig_fi = plot_feature_importance(final_model, processed_feature_names, selected_model_type)
                 if fig_fi: plot_col2.pyplot(fig_fi, clear_figure=True); plot_col2.caption("Feature Importance (Final Model)")
                 else: plot_col2.info("Feature importance not available.")
            except Exception as fi_e: st.warning(f"Could not generate FI plot: {fi_e}")

            if task_type == "Classification": #ROC Curve plot 
                 classes_roc = st.session_state.get('label_classes', np.unique(y_original))
                 fig_roc = plot_roc_curve(y_test_final, y_pred_proba_test, classes_roc)
                 if fig_roc: st.pyplot(fig_roc, clear_figure=True); st.caption("ROC Curve (Test Set)")

            #Prepare final model to download (I'm using plk, but joblib could also be used, although its heavier)
            st.subheader("💾 Prepare Final Model for Download")
            try:
                model_bytes_io = io.BytesIO(); joblib.dump(pipeline, model_bytes_io); model_bytes_io.seek(0)
                st.session_state.model_bytes = model_bytes_io.getvalue(); st.session_state.last_model_type = selected_model_type
                st.info("Final model ready. Download button appears below.")
            except Exception as dump_error: st.error(f"❌ Error preparing model download: {dump_error}"); st.session_state.model_bytes = None

            #Add run to history
            run_entry = {
                "timestamp": datetime.now(),
                "config": run_config,
                "results": run_results
            }
            st.session_state.training_history.append(run_entry)
            st.success("Run details saved to history in the sidebar.")

        #Catch all erros during training
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {e}"); st.error("Traceback:"); st.code(traceback.format_exc())
            st.session_state.model_bytes = None; st.session_state.last_model_type = None;
            
#Display the button of "Download"
if st.session_state.get('model_bytes') is not None:
    st.divider(); st.header("💾 Download Trained Model")
    st.download_button(label=f"Download {st.session_state.last_model_type} Pipeline (.pkl)", data=st.session_state.model_bytes, file_name=f"{st.session_state.last_model_type.replace(' ', '_').lower()}_pipeline.pkl", mime="application/octet-stream", key="download_model_button")
    if st.button("Clear Current Results & Model", key="clear_results_button"):
        st.session_state.model_bytes = None; st.session_state.label_classes = None; st.session_state.last_model_type = None;
        st.rerun()

#Handle the cases where the dataset is not loaded
elif df is None and not error_message:
    st.info("⬅️ Select or upload data using the sidebar.")
    st.session_state.model_bytes = None; st.session_state.last_model_type = None; st.session_state.label_classes = None;

#Footer of the streamlit application
st.sidebar.markdown("---")
st.sidebar.info("AI Model Trainer - Assignment (by: Víctor Pérez)")
