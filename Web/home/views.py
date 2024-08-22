import joblib
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
import pickle
import numpy as np
from datetime import datetime

# Reference date for conversion
reference_date = pd.to_datetime('1924-02-17')

# Default values dictionary
default_values_f1 = {
    'activation_date': 35306.570183,
    'gender_cv_id': 2719.321740,
    'date_of_birth': 18133.860361,
    'legal_form_enum': 1.003603,
    'principal_amount': 15586.604141,
    'interest_period_frequency_enum': 2.509243,
    'interest_method_enum': 0.970859,
    'interest_calculated_in_period_enum': 0.029141,
    'approvedon_date': 35274.502584,
    'expected_disbursedon_date': 35274.502589,
    'disbursedon_date': 35274.539349,
    'expected_maturedon_date': 35957.505120,
    'maturedon_date': 35784.309945,
    'transaction_type_enum': 1.763809,
    'transaction_date': 35519.251317,
    'amount': 4381.822410,
    'submitted_on_date': 36389.967922
}

default_values_f2 = {
    'activation_date': 35306.570183,
    'gender_cv_id': 2719.321740,
    'date_of_birth': 18133.860361,
    'validatedon_date': 35296.196238,
    'legal_form_enum': 1.003603,
    'principal_amount': 15586.604141,
    'nominal_interest_rate_per_period': 1.413549,
    'interest_period_frequency_enum': 2.509243,
    'annual_nominal_interest_rate': 1.413549,
    'interest_method_enum': 0.970859,
    'interest_calculated_in_period_enum': 0.029141,
    'term_frequency': 22.412665,
    'number_of_repayments': 22.412665,
    'approvedon_date': 35274.502584,
    'expected_disbursedon_date': 35274.502589,
    'disbursedon_date': 35274.539349,
    'expected_maturedon_date': 35957.505120,
    'maturedon_date': 35784.309945,
    'transaction_type_enum': 1.763809,
    'transaction_date': 35519.251317,
    'amount': 4381.822410,
    'submitted_on_date': 36389.967922,
    'created_date': 36389.967936
}

# Function to load models when needed
def load_models():
    model_f1_1 = pickle.load(open(os.path.join(settings.BASE_DIR, 'Nominal_models/linear_regression_model_new.pkl'), 'rb'))
    model_f1_2 = pickle.load(open(os.path.join(settings.BASE_DIR, 'Nominal_models/decision_tree_regressor_model_new.pkl'), 'rb'))
    model_f1_3 = pickle.load(open(os.path.join(settings.BASE_DIR, 'Nominal_models/random_forest_regressor_model_new.pkl'), 'rb'))

    model_f2_1 = pickle.load(open(os.path.join(settings.BASE_DIR, 'Intrest_model/decision_tree_model.pkl'), 'rb'))
    model_f2_2 = pickle.load(open(os.path.join(settings.BASE_DIR, 'Intrest_model/logistic_model.pkl'), 'rb'))
    model_f2_3 = pickle.load(open(os.path.join(settings.BASE_DIR, 'Intrest_model/xgb_model.pkl'), 'rb'))

    return model_f1_1, model_f1_2, model_f1_3, model_f2_1, model_f2_2, model_f2_3

# Function to parse and convert date string to difference in days from reference_date
def convert_date_string_to_difference(date_input):
    try:
        if isinstance(date_input, float) or date_input.isdigit():
            return float(date_input)

        date_obj = pd.to_datetime(date_input)
        return (date_obj - reference_date).days
    except Exception as e:
        print(f"Error parsing date: {e}")
        return 0

def index(request):
    model_f1_1, model_f1_2, model_f1_3, model_f2_1, model_f2_2, model_f2_3 = load_models()

    models_f1 = ['Linear_regression', 'Decision_tree', 'Random_forest']
    models_f2 = ['Decision_tree', 'Logistic', 'Xgb']

    features_f1 = [
    'activation_date', 'gender_cv_id', 'date_of_birth', 'legal_form_enum',
    'principal_amount', 'interest_period_frequency_enum', 
    'interest_method_enum', 'interest_calculated_in_period_enum', 
    'approvedon_date', 'expected_disbursedon_date', 'disbursedon_date',
    'expected_maturedon_date', 'maturedon_date', 'transaction_type_enum', 
    'transaction_date', 'amount', 'submitted_on_date'
    ]

    features_f2 = [
        'activation_date', 'gender_cv_id', 'date_of_birth', 'validatedon_date',
        'legal_form_enum', 'principal_amount', 'nominal_interest_rate_per_period',
        'interest_period_frequency_enum', 'annual_nominal_interest_rate',
        'interest_method_enum', 'interest_calculated_in_period_enum', 'term_frequency',
        'number_of_repayments', 'approvedon_date', 'expected_disbursedon_date',
        'disbursedon_date', 'expected_maturedon_date', 'maturedon_date',
        'transaction_type_enum', 'transaction_date', 'amount', 'submitted_on_date',
        'created_date'
    ]

    selected_features = request.session.get('selected_features', {})
    selected_feature_set = request.session.get('selected_feature_set', 'F1')

    if request.method == 'POST':
        action = request.POST.get('action', '')
        if action == 'select':
            feature_set = request.POST.get('feature_set', 'F1')
            request.session['selected_feature_set'] = feature_set
        elif action == 'submit':
            selected_features_list = request.POST.getlist('selected_features')
            form_data = {}

            # Use default values for missing features
            if selected_feature_set == 'F1':
                default_values = default_values_f1
            else:
                default_values = default_values_f2

            for feature in default_values.keys():
                # Check if the feature is selected by the user, otherwise use default
                if feature in selected_features_list:
                    # If 'date' in feature, convert date input
                    if 'date' in feature:
                        form_data[feature] = [convert_date_string_to_difference(request.POST.get(feature, ''))]
                    else:
                        form_data[feature] = [request.POST.get(feature, default_values[feature])]
                else:
                    form_data[feature] = [default_values[feature]]

            feature_df = pd.DataFrame(form_data).astype(float)
            selected_model = request.POST.get('model', '')

            if selected_feature_set == 'F1':
                prediction = predict_f1(feature_df, selected_model, model_f1_1, model_f1_2, model_f1_3)
            elif selected_feature_set == 'F2':
                prediction = predict_f2(feature_df, selected_model, model_f2_1, model_f2_2, model_f2_3)
            else:
                prediction = "Invalid Feature Set"

            request.session['selected_features'] = {feature: '' for feature in selected_features_list}

            return render(request, 'home/index.html', {
                'models': models_f1 if selected_feature_set == 'F1' else models_f2,
                'all_features': features_f1 if selected_feature_set == 'F1' else features_f2,
                'selected_features': selected_features,
                'selected_feature_set': selected_feature_set,
                'prediction_result': prediction,
            })

    return render(request, 'home/index.html', {
        'models': models_f1 if selected_feature_set == 'F1' else models_f2,
        'all_features': features_f1 if selected_feature_set == 'F1' else features_f2,
        'selected_features': selected_features,
        'selected_feature_set': selected_feature_set
    })

def predict_f1(features_df, selected_model, model_f1_1, model_f1_2, model_f1_3):

    if selected_model == 'Linear_regression':
        kt = model_f1_1.predict(features_df)[0]
        if(kt < 0):
            return "Invalid, try other model"
        else:
            return str(kt)
    elif selected_model == 'Decision_tree':
        st = str(model_f1_2.predict(features_df)[0])
        return st                       
    elif selected_model == 'Random_forest':
        qt = str(model_f1_3.predict(features_df)[0])
        return qt
    else:
        return "Invalid model selection for F1"

def predict_f2(features_df, selected_model, model_f2_1, model_f2_2, model_f2_3):
    features_df = features_df.astype(float)
    features_df = np.array(features_df).reshape(1, -1)
    if selected_model == 'Decision_tree':
        m = model_f2_1.predict(features_df)
        m = str(m)
        return m
    elif selected_model == 'Logistic':
        l =  model_f2_2.predict(features_df)
        l = str(l)
        return l
    elif selected_model == 'Xgb':
        k = model_f2_3.predict(features_df)
        k = str(k)
        return k
    else:
        return "Invalid model selection for F2"
