# ------------------------------
# Standard library
import sys
import importlib
import time
import io
import zipfile

# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap
import pickle

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Custom modules
sys.path.append("..")
import utils.utils as utils 
import utils.models as models
import utils.AG_FTIR as ag
import utils.preprocessing as pp

# Initialization
shap.initjs()
importlib.reload(utils)

#Keras
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

# Random seed for reproducibility
import random as python_random
from numpy.random import seed
seed(0)
np.random.seed(123)
python_random.seed(123)
# ------------------------------

# ---------------------------------------------------------------------------------------------------------
#BIOFTIR
# ---------------------------------------------------------------------------------------------------------
st.logo(image="../images/bioftir_logo_small.png")

st.image(image="../images/bioftir_image.png", caption="An AutoML made for FTIR Biological Data.", width=400)
st.divider()

st.markdown("***Training and Test data***")
train_file = st.file_uploader('Training data', type=["csv"])

if train_file is not None:
    train_val_data = pd.read_csv(train_file)
    wavenumber = train_val_data.iloc[:,:-1].columns

    try:
        start_, end_ = st.select_slider(
            "Wavelength interval range to analize (Biodigital impression 1800-900)",
            options=list(wavenumber.astype(str)),
            value=(wavenumber.astype(str)[utils.get_index(train_val_data.iloc[:,:-1],1800)],
                    wavenumber.astype(str)[utils.get_index(train_val_data.iloc[:,:-1],900)]),
        )
    except:
        start_, end_ = st.select_slider(
            "Select a range of wavelength",
            options=list(wavenumber.astype(str)),
            value=(wavenumber.astype(str)[0], wavenumber.astype(str)[-1]),
        )
    #Range values for concatenation
    try:
        default_values = [
            wavenumber.astype(str)[utils.get_index(train_val_data.iloc[:, :-1], 3050)], #Lipidic region
            wavenumber.astype(str)[utils.get_index(train_val_data.iloc[:, :-1], 2800)]  #Lipidic region
        ]
    except Exception:
        default_values = []
        
    options_concat = st.multiselect(
        "What ranges would you like to concatenate?",
        list(wavenumber.astype(str)),
        default=default_values,
    )
    ranges = sorted(set([start_, end_]+options_concat), key=float, reverse=True)
    st.write("You selected:", ranges) 

else:
    st.warning("you need to upload a csv or excel file.")

# test_data input
test_file = st.file_uploader('test data', type=["csv"])
if test_file is not None:
    test_data = pd.read_csv(test_file)

# Sidebar options
with st.sidebar:
    #Model selection
    st.subheader("***Models***")
    
    options = ["KNN", "LDA", "SVM", "XGBoost", "BiLSTM"]
    default_selection = ["KNN"]
    model_selection = st.multiselect("Select the models:", options, default=default_selection)

    model_number = st.number_input("Insert a number for models quantitity", value=1)
    st.markdown(f"Selected models: {model_selection} x {model_number}")

    #Pre-processing
    st.divider()
    st.subheader("***Pre-processing***")
    outlier_options = ["Mahalanobis", "Isolation_Forest"]
    outlier_detection = st.pills("Outlier detection method", outlier_options, selection_mode="single", default=[])

    base_options = ["Original", "PCA", "MNF", "SMNF"]
    base_selection = st.pills("Include method", base_options, selection_mode="multi", default=["Original", "PCA", "MNF"])

    #Hyper-parameters
    st.divider()
    
    header_left, header_right = st.columns([1,10], vertical_alignment="top")
    header_right.subheader("***Hyperparameters tunning***")
    include_tunning = header_left.toggle("", value=True, key='include_tunning')

    left, right = st.columns(2, vertical_alignment="top")
    
    n_iter = left.number_input("n_iter", value=1000)
    random_state_value = right.number_input('random_state', min_value=0, value=None)
    refit = right.selectbox(
        "Refit options",
        ("roc_auc", "accuracy", "f1_score"),
    )
    es_bilstm = left.number_input('es_bilstm', value=15)
    

    #AG options
    st.divider()

    header_left, header_right = st.columns([1,10], vertical_alignment="top")
    header_right.subheader("***AG Options***")
    include_ga= header_left.toggle("", value=True, key='include_ga')
    # st.subheader("***AG Options***")
    # include_ga = st.checkbox("", value=True, key='include_ga')
    left, middle, right = st.columns(3, vertical_alignment="top")
    ag_generations = left.number_input("Generations", value=100)
    ag_earlystopping = middle.number_input("Early-stopping", value=5)
    ag_tp_crossover = right.number_input("Population", value=50)
    ag_n_pairs= left.number_input("X-over n_pairs", value=15)
    ag_random_prob= middle.number_input("Gen random", value=True)
    ag_cv= right.number_input("Fitness CV", value=5)

    #XAI Options
    st.divider()
    st.subheader("***Explainable Artificial Intelligence (XAI)***")

    left, right = st.columns(2, vertical_alignment="top")
    xai_knn_n_samples = left.number_input("N_samples", value=10)

# Load trainned model and Run    
left, middle, right= st.columns(3, vertical_alignment="top")
run = False
if (left.button("Run", type="primary", use_container_width=True)) and train_file is not None:
    run = True

loaded_model = None
if test_file is not None:
    loaded_model = right.file_uploader('Load trainned model', type=["pkl"])

if (middle.button("Run loaded Model", type="secondary", use_container_width=True)) and test_file is not None and loaded_model is not None:
    run = True

#Run Experiments
# ---------------------------------------------------------------------------------------------------------
if run:
    if train_file is not None and loaded_model is None:

        train_val_classes = train_val_data.iloc[:,-1].values
        train_val_data = train_val_data.iloc[:,:-1]
        wavenumber = train_val_data.columns

        #Outlier Detection Method to apply
        if outlier_detection:
            if outlier_detection == "Mahalanobis":
                train_val_data, mask = pp.detect_outliers_mahalanobis(train_val_data)
            else:
                train_val_data, mask = pp.detect_outliers_isolation_forest(train_val_data)
            train_val_classes = train_val_classes[mask]

        if test_file is not None:
            test_classes = test_data.iloc[:,-1].values
            test_data = test_data.iloc[:,:-1]

            if outlier_detection:
                if outlier_detection == "Mahalanobis":
                    test_data, _ = pp.detect_outliers_mahalanobis(test_data)
                else:
                    test_data, _ = pp.detect_outliers_isolation_forest(test_data)
                test_classes = test_classes[mask]
        else:
            st.write('We will split your training data into train/test data since there was no test data uploaded')
            
            train_val_data, test_data, train_val_classes, test_classes = train_test_split(
                train_val_data, train_val_classes, test_size=0.204, stratify=train_val_classes, random_state=42
            )

        st.write(f'Training data {train_val_data.shape}:')
        train_val_data
        st.write(f'Test data {test_data.shape}:')
        test_data   

        st.divider()
        

        #Data description
        with st.spinner("Obtaining data information...", show_time=True):

            with st.spinner("Obtaining Training data information...", show_time=True):
                st.subheader('**Training data description**:')
                
                st.markdown('**Data plot**')
                st.pyplot(utils.plot_spectrum_streamlit([train_val_data[train_val_classes==1], train_val_data[train_val_classes==0]],label=['Positive', 'Negative'],color=['red','green'], alpha=0.1, figsize=(15,8)))
                st.markdown('**Data plot - Mean values**')
                st.pyplot(utils.plot_spectrum_streamlit([pd.DataFrame(train_val_data[train_val_classes==1].mean()).T, pd.DataFrame(train_val_data[train_val_classes==0].mean()).T],label=['Positive', 'Negative'],color=['red','green'], alpha=1, figsize=(15,8)))

                st.markdown('**Descriptive summary statistics**')
                st.dataframe(train_val_data.describe())

                pca_df = utils.visual_pca_plot_streamlit(train_val_data)
                pca_df['class'] = train_val_classes.astype(int)
                chart = utils.visual_pca_colorplot_streamlit(pca_df, 'PCA 2 components')
                st.altair_chart(chart, use_container_width=True)

            #Test Data description
            with st.spinner("Obtaining Test data information...", show_time=True):
                st.subheader('**Test data description**:')
                
                st.markdown('**Data plot**')
                st.pyplot(utils.plot_spectrum_streamlit([test_data[test_classes==1], test_data[test_classes==0]],label=['Positive', 'Negative'],color=['red','green'], alpha=0.1, figsize=(15,8)))
                st.markdown('**Data plot - Mean values**')
                st.pyplot(utils.plot_spectrum_streamlit([pd.DataFrame(test_data[test_classes==1].mean()).T, pd.DataFrame(test_data[test_classes==0].mean()).T],label=['Positive', 'Negative'],color=['red','green'], alpha=1, figsize=(15,8)))

                st.markdown('**Descriptive summary statistics**')
                st.dataframe(test_data.describe())

                pca_df = utils.visual_pca_plot_streamlit(test_data)
                pca_df['class'] = test_classes.astype(int)
                chart = utils.visual_pca_colorplot_streamlit(pca_df, 'PCA 2 components')
                st.altair_chart(chart, use_container_width=True)

            #Determining region intervals
            ranges = [ float(x) for x in ranges]
            region = [[(ranges[i], ranges[i+1]) for i in range(0, len(ranges)-1, 2)]]

            
        
        #Pre-processing
        #Amida I normalization, Savitzky-Golay filter and derivations
        with st.spinner("Normalizing data...", show_time=True):
 
            X_normalized, X_savgol, X_savgol_df, X_savgol_df_2 = pp.apply_preprocessing(train_val_data, train_val_classes, region)
            X_test_normalized, X_test_savgol, X_test_savgol_df, X_test_savgol_df_2 = pp.apply_preprocessing(test_data, test_classes, region)

            # #Spectrum concatenation (when choosen more than one interval)
            ori_normalized = pp.conjugate_bases(X_normalized, X_test_normalized, 'Amida I')
            ori_savgol = pp.conjugate_bases(X_savgol, X_test_savgol, 'Savgol')
            ori_savgol_df = pp.conjugate_bases(X_savgol_df, X_test_savgol_df, 'Savgol_df')
            ori_savgol_df_2 = pp.conjugate_bases(X_savgol_df_2, X_test_savgol_df_2, 'Savgol_df2')

            ori_bases = []
            if "Original" in base_selection:
                ori_bases = [ori_normalized, ori_savgol, ori_savgol_df, ori_savgol_df_2]

        st.badge("Normalized", icon=":material/check:", color="green")

        #PCA application
        pca_bases = []
        if "PCA" in base_selection:
            with st.spinner("Applying PCA on data...", show_time=True):
                pca_normalized = pp.pca_multi_transform(X_normalized, X_test_normalized)
                pca_savgol = pp.pca_multi_transform(X_savgol, X_test_savgol)
                pca_savgol_df = pp.pca_multi_transform(X_savgol_df, X_test_savgol_df)
                pca_savgol_df_2 = pp.pca_multi_transform(X_savgol_df_2, X_test_savgol_df_2)

                pca_bases = [pca_normalized, pca_savgol, pca_savgol_df, pca_savgol_df_2]

            st.badge("PCA applied", icon=":material/check:", color="green")

        #MNF application
        mnf_bases = []
        if "MNF" in base_selection:
            with st.spinner("Applying MNF on data...", show_time=True):
                my_bar = st.progress(0, text="Please wait...")

                mnf_normalized = pp.compute_mnf(X_normalized, X_test_normalized)
                my_bar.progress(25, text="Just a little more...")
                mnf_savgol = pp.compute_mnf(X_savgol, X_test_savgol)
                my_bar.progress(50, text="Just a bit more...")
                mnf_savgol_df = pp.compute_mnf(X_savgol_df, X_test_savgol_df)
                my_bar.progress(75, text="Almost there...")
                mnf_savgol_df_2 = pp.compute_mnf(X_savgol_df_2, X_test_savgol_df_2)
                my_bar.progress(100, text="Here we go...")

                mnf_bases = [mnf_normalized, mnf_savgol, mnf_savgol_df, mnf_savgol_df_2]
                my_bar.empty()
            st.badge("MNF applied", icon=":material/check:", color="green")  

        #New method SMNF application
        smnf_bases = []
        if "SMNF" in base_selection:
            with st.spinner("Applying Strategic MNF on data...", show_time=True):
                my_bar = st.progress(0, text="Please wait...")
                smnf_normalized = pp.compute_smnf(X_normalized, X_test_normalized, segments=1)
                my_bar.progress(25, text="Just a little more...")
                smnf_savgol = pp.compute_smnf(X_savgol, X_test_savgol, segments=1)
                my_bar.progress(50, text="Just a bit more...")
                smnf_savgol_df = pp.compute_smnf(X_savgol_df, X_test_savgol_df, segments=1)
                my_bar.progress(75, text="Almost there...")
                smnf_savgol_df_2 = pp.compute_smnf(X_savgol_df_2, X_test_savgol_df_2, segments=1)
                my_bar.progress(100, text="Here we go...")

                smnf_bases = [smnf_normalized, smnf_savgol, smnf_savgol_df, smnf_savgol_df_2]
                my_bar.empty()
            st.badge("SMNF applied", icon=":material/check:", color="green")  

        #Models Selected
        with st.spinner("Getting input information...", show_time=True):
            
            xgb_model = models.xgb_model()
            lda_model = models.lda_model()
            knn_model = models.knn_model()
            svc_model = models.svc_model()
            bilstm_model = models.bilstm_model()

            model_dict = {
                'KNN': knn_model,
                'LDA': lda_model,
                'SVM': svc_model,
                'XGBoost': xgb_model,
                'BiLSTM': bilstm_model
            }

        #Loop for models selected
        for i_number in range(model_number):
            train_models = []
            for m in model_selection:
                train_models.append(model_dict[m])
        
            #sets de random_state value if not specified
            if random_state_value is None:
                random_state = np.random.randint(0,10000)
            else:
                random_state = random_state_value

            with st.spinner("Models...", show_time=True):
                best_tunned = []
                for model in train_models:

                    pca_choosen = []
                    for i in range(len(pca_bases)):
                        for base in pca_bases[i]:
                            pca_choosen.append(pp.pca_choices(xgb_model, [base], train_val_classes))
                    mnf_choosen = []
                    for i in range(len(mnf_bases)):
                        for base in mnf_bases[i]:
                            mnf_choosen.append(pp.pca_choices(xgb_model, [base], train_val_classes))
                    smnf_choosen = []
                    for i in range(len(smnf_bases)):
                        for base in smnf_bases[i]:
                            smnf_choosen.append(pp.pca_choices(xgb_model, [base], train_val_classes))

                    ori_flat = []
                    for i in range(len(ori_bases)):
                        for j in range(len(ori_bases[i])):
                            ori_flat.append(ori_bases[i][j])

                    complete_base = ori_flat + pca_choosen + mnf_choosen + smnf_choosen
                    
                    #Hyperparameters tunning
                    if include_tunning:
                        my_bar = st.progress(0, text="Tunning hyperparameters. Please wait.")
                        complete_optimized = []
                        for i in range(len(complete_base)):
                            my_bar.progress((i+1)/len(complete_base), text="Tunning hyperparameters. Please wait.")
                            clas_calc = models.optmize_model(complete_base[i][0], train_val_classes, model, n_splits=10, n_iter=n_iter, n_repeats=1, early_stopping_rounds=es_bilstm , random_state=random_state)
                            complete_optimized.append(clas_calc)
                        my_bar.empty()
                    else:
                        with st.spinner("Configuring hyperparameters...", show_time=True):
                            complete_optimized = []
                            for i in range(len(complete_base)):
                                clas_calc = models.optmize_model(complete_base[i][0], train_val_classes, model, no_tune=True, n_splits=2, n_iter=1, n_repeats=1, early_stopping_rounds=es_bilstm , random_state=random_state)
                                complete_optimized.append(clas_calc)
      
                    solution_base = []
                    solution = [] 
                    solution_value = 0  
                    for i in range(len(complete_optimized)):
                        if complete_optimized[i].best_score_ > solution_value:
                            solution_value = complete_optimized[i].best_score_ 
                            scored_base, model_tunned = (complete_base[i], complete_optimized[i])
                    
                    best_tunned.append((model, scored_base, model_tunned))
            if include_tunning:
                st.badge("Models tunned", icon=":material/check:", color="green")
            
            #Computing the best results for each model
            if include_ga:
                with st.spinner("Selecting features with Genetic algorithm...", show_time=True):
                        
                    best_results = []
                    for best in best_tunned:
                        
                        pop_cal = ag.ag_pipeline(best[1][0], train_val_classes, model=best[2], generations=ag_generations,Tp=ag_tp_crossover,
                                                    gen_random_prob=ag_random_prob, pairs_number=ag_n_pairs, cv=ag_cv, early_stopping=ag_earlystopping)
                        
                        solution_base, solution = (best[1], pop_cal[0])
                        best_results.append((best[2], solution_base, solution))

                    # best_results = [(bilstm_model, complete_base[0], (np.random.choice(2,complete_base[0][0].shape[1]),0.5))]
                st.badge("GA applied", icon=":material/check:", color="green")
            else:
                with st.spinner("Colecting solutions...", show_time=True):
                    best_results = []
                    for best in best_tunned:
                        solution_base, solution = (best[1], (np.random.randint(1,2,best[1][0].shape[1]), -1))
                        best_results.append((best[2], solution_base, solution))

            #Best solutions for each model
            #--------------------------------------------------------------------------------------------------------------------#
            for i_model in range(len(best_results)):

                best_model, solution_base, solution = best_results[i_model][0], best_results[i_model][1], best_results[i_model][2]

                st.divider()
                st.subheader('***Best model information:***')
                st.markdown(f'***Model***: {type(best_model.best_estimator_)}')
                st.markdown(f'***Data***: {solution_base[2]} - ***Fitness***: {solution[1]} - ***Random_state***: {random_state}')

                mask = np.array(solution[0], dtype=bool)

                X_selected = solution_base[0].iloc[:, mask]
                test_selected = solution_base[1].iloc[:, mask]

                test_base = test_selected.values
                X = X_selected.values
                y = train_val_classes.astype(np.float64)

                st.markdown(f'***Hyperparameters optimization***: {best_model.best_params_}')
                st.markdown(f'***Refit***: {refit}')
                models.c_report_streamlit(best_model)
                report_save = best_model

                if isinstance(best_model.best_estimator_, KerasClassifier):
                    tf.keras.backend.clear_session()
                    timesteps = 1
                    X = X.reshape((X.shape[0], timesteps, X.shape[1]))
                    test_base = test_base.reshape((test_base.shape[0], timesteps, test_base.shape[1]))

                    timesteps = 1
                    params = best_model.best_params_
                    es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=30, restore_best_weights=True)

                    best_model = KerasClassifier(
                        model=models.build_bilstm_model,
                        **params,
                        model__input_shape=(timesteps, X.shape[2]),
                        epochs=10000,
                        verbose=0,
                        callbacks=[es],
                    )
                    best_model.fit(X,y)

                else:
                    best_model = best_model.best_estimator_.fit(X,y)

                y_pred = best_model.predict(test_base)
                models.c_matrix_streamlit(true_classes=test_classes.astype(np.float64), y_pred=y_pred)

                with st.spinner("Generating model explanation...", show_time=True):
                    if xai_knn_n_samples > 0:
                        if isinstance(best_model, type(knn_model)) or isinstance(best_model, type(svc_model)) or isinstance(best_model, type(lda_model)):
                            
                            X_df = pd.DataFrame(X) 
                            background = X_df.sample(n=30, random_state=42)
                            instances = pd.DataFrame(test_base[:xai_knn_n_samples, :], columns=X_selected.columns)

                            explainer = shap.KernelExplainer(best_model.predict_proba, background)
                            shap_values = explainer.shap_values(instances)
                            shap_values_transposed = np.transpose(shap_values, (2, 0, 1))

                            plt.figure()
                            shap.summary_plot(shap_values_transposed[0], features=instances, feature_names=instances.columns, show=False)
                            st.pyplot(plt.gcf())
                            plt.clf()

                        if isinstance(best_model, type(xgb_model)):
                            explainer = shap.Explainer(best_model)
                            shap_values = explainer(test_base)
                            plt.figure()
                            shap.summary_plot(shap_values, test_base, feature_names=X_selected.columns)
                            st.pyplot(plt.gcf())
                            plt.clf()

                        if isinstance(best_model, KerasClassifier):
                            best_model = best_model.model_
                            background = test_base[np.random.choice(test_base.shape[0], 10, replace=False)]
                            explainer = shap.GradientExplainer(best_model, background)
                            shap_values = explainer.shap_values(test_base)

                            re = np.squeeze(shap_values, axis=-1)
                            mean_shap = re.mean(axis=1)

                            plt.figure()
                            shap.summary_plot(mean_shap, features=test_base[:, 0, :], feature_names=test_selected.columns, show=False)
                            st.pyplot(plt.gcf())
                            plt.clf()

                #PCA Explanation
                try:
                    pca_loadings, heatmap_buffer = models.xai_pca_streamlit(pca=solution_base[3], columns=solution_base[4], metadata=solution_base[2], n_contributions=solution_base[0].shape[1], n_components=10,plot=True)
                except:
                    pass

                #Model and data preparation for download
                tv_df = X_selected
                tv_df['classes'] = train_val_classes

                t_data = test_selected
                t_data['classes'] = test_classes

                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    model_buffer = io.BytesIO()
                    pickle.dump(best_model, model_buffer)
                    model_buffer.seek(0)

                    zf.writestr("model.pkl", model_buffer.read()) 

                    try:
                        #heatmap
                        if heatmap_buffer is not None:
                            zf.writestr("pca_heatmap.png", heatmap_buffer.read())

                        pca_buffer = io.BytesIO()
                        pickle.dump(solution_base[3], pca_buffer)
                        pca_buffer.seek(0)
                        zf.writestr("pca", pca_buffer.read()) 

                        csv_buffer = io.StringIO()
                        pca_loadings.to_csv(
                            csv_buffer,
                            sep=',',
                            index=True,
                            line_terminator='\n'
                        )
                        csv_buffer.seek(0)
                        zf.writestr("pca_loadings.csv", csv_buffer.read())
                    except:
                        pass

                    tv_csv = tv_df.to_csv(index=False).encode('utf-8')
                    zf.writestr("train_val_data.csv", tv_csv)

                    t_csv = t_data.to_csv(index=False).encode('utf-8')
                    zf.writestr("test_data.csv", t_csv)

                    #Classification Report
                    best_index = report_save.best_index_
                    cv = report_save.cv_results_
                    metrics = [
                        ("AUC-ROC",        "roc_auc"),
                        ("Acurácia",       "accuracy"),
                        ("Precisão",       "precision"),
                        ("Especificidade", "specificity"),
                        ("Sensibilidade",  "recall"),
                        ("F1-score",       "f1_score"),
                    ]

                    lines = ["--- Classification Report (CV) Train-Validation ---"]
                    for label, met in metrics:
                        mean = cv[f"mean_test_{met}"][best_index]
                        std  = cv[f"std_test_{met}"][best_index]
                        lines.append(f"{label}: {mean:.3f} ±{std:.3f}")
                    lines.append("---")
                    lines.append("")
                    lines.append("Best model information:\n" + 
                                  f'Data: {solution_base[2]} - Fitness: {solution[1]} - Random_state: {random_state}')

                    zf.writestr("cv_report.txt", "\n".join(lines))
                buffer.seek(0)

                st.download_button(
                    label=f"Save model and data (f'{i_model}|{i_number}')",
                    data=buffer,
                    file_name="model_and_data.zip",
                    mime="application/zip"
                )

    #Loaded models and test files
    elif test_file is not None and loaded_model is not None:

        test_classes = test_data.iloc[:,-1].values
        test_data = test_data.iloc[:,:-1]
        wavenumber = test_data.columns

        st.subheader('Test data', divider="orange")
        test_data   

        with st.spinner("Obtaining data information...", show_time=True):
            st.subheader('**Test data description**:')
            st.markdown('**Data plot**')
            st.pyplot(utils.plot_spectrum_streamlit([test_data[test_classes==1], test_data[test_classes==0]],label=['Positive', 'Negative'],color=['red','green'], alpha=0.1, figsize=(15,8)))
            st.markdown('**Data describe**')
            st.dataframe(test_data.describe())
            st.markdown(f'**Data shape**: {test_data.shape}')

        st.divider()

        test_base = test_data.values
        classifier_sv1 = pickle.load(loaded_model)

        if isinstance(classifier_sv1, (KerasClassifier, Sequential)):
            test_base = test_base.reshape((test_base.shape[0], 1, test_base.shape[1]))
        
        st.write(f"Model type: {type(classifier_sv1)}")
       
        y_pred = classifier_sv1.predict(test_base)

        models.c_matrix_streamlit(true_classes=test_classes.astype(np.float64), y_pred=(y_pred >= 0.5).astype(int))

        with st.spinner("Generating model explanation...", show_time=True):
            
            if isinstance(classifier_sv1, (KNeighborsClassifier, SVC, LinearDiscriminantAnalysis)):
                
                X_df = pd.DataFrame(test_base) 
                background = X_df.sample(n=30, random_state=42)
                instances = pd.DataFrame(test_base[:xai_knn_n_samples, :], columns=wavenumber)

                explainer = shap.KernelExplainer(classifier_sv1.predict_proba, background)
                shap_values = explainer.shap_values(instances)
                shap_values_transposed = np.transpose(shap_values, (2, 0, 1))

                plt.figure()
                shap.summary_plot(shap_values_transposed[0], features=instances, feature_names=instances.columns, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            if isinstance(classifier_sv1, type(models.xgb_model())):
                explainer = shap.Explainer(classifier_sv1)
                shap_values = explainer(test_base)
                plt.figure()
                # shap.summary_plot(shap_values, test_base, feature_names=["PC" + str(int(col)+1) for col in wavenumber])
                shap.summary_plot(shap_values, test_base, feature_names=wavenumber)
                st.pyplot(plt.gcf())
                plt.clf()

            if isinstance(classifier_sv1, (KerasClassifier, Sequential)):
                best_model = classifier_sv1
                background = test_base[np.random.choice(test_base.shape[0], 10, replace=False)]
                explainer = shap.GradientExplainer(best_model, background)
                shap_values = explainer.shap_values(test_base)

                re = np.squeeze(shap_values, axis=-1)
                mean_shap = re.mean(axis=1)

                plt.figure()
                shap.summary_plot(mean_shap, features=test_base[:, 0, :], feature_names=test_data.columns, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

        #--------------------------------------------------------------------------------------------------------------------#


