import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def load_data(filepath):
    return pd.read_csv(filepath, sep='|')

def remove_redundant_columns(df, columns):
    return df.drop(columns=columns)

def remove_duplicates(df, subset_column):
    return df.drop_duplicates(subset=[subset_column])

def replace_values(df, old_value, new_value):
    df.replace(old_value, new_value, inplace=True)

def drop_columns_by_missing_percentage(df, threshold):
    missing_values_percent = df.isnull().mean() * 100
    columns_to_drop = missing_values_percent[missing_values_percent > threshold].index
    return df.drop(columns=columns_to_drop)

def save_to_csv(df, filename, separator='|'):
    df.to_csv(filename, sep=separator, index=False)

def process_date_columns(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    min_date = df[date_columns].min().min()
    reference_date = pd.to_datetime(min_date)
    for col in date_columns:
        df[col] = df[col].fillna(reference_date)
        df[col] = (df[col] - reference_date).dt.days
    return df

def convert_to_categorical(df, categorical_columns):
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

def remove_columns_with_missing_values(df, missing_threshold=0):
    missing_values = df.isnull().mean() * 100
    columns_to_remove = missing_values[missing_values > missing_threshold].index
    print(f"Number of columns with missing values above {missing_threshold}%: {len(columns_to_remove)}")
    if len(columns_to_remove) > 0:
        print(f"Columns removed due to missing values: {columns_to_remove}")
    return df.drop(columns=columns_to_remove)

def clean_data(filepath):
    df = load_data(filepath)
    
    redundant_columns = [
        'id_client', 'id_loan', 'loan_status_id', 'days_in_month_enum', 'days_in_year_enum', 
        'loan_id', 'payment_detail_id', 'appuser_id', 'id', 'id_client', 'client_id', 'product_id', 
        'fund_id', 'currency_multiplesof', 'submittedon_userid', 'approved_principal', 'currency_multiplesof', 
        'submittedon_userid', 'approvedon_userid', 'disbursedon_userid', 'closedon_userid', 'rejectedon_userid', 
        'loan_product_counter', 'version', 'is_equal_amortization'
    ]
    df = remove_redundant_columns(df, redundant_columns)
    df = remove_duplicates(df, 'loan_id')
    replace_values(df, '\\N', np.nan)
    
    df = drop_columns_by_missing_percentage(df, 50)
    
    return df

def perform_chi_square_test(df, continuous_target, categorical_features, num_bins=10, significance_level=0.01):
    # Discretize the continuous target variable
    df[continuous_target + '_binned'] = pd.qcut(df[continuous_target], q=num_bins, labels=False, duplicates='drop')
    
    # Perform the Chi-Square Test
    chi2_scores, p_values = chi2(df[categorical_features], df[continuous_target + '_binned'])
    
    # Create a DataFrame for the results
    chi2_results = pd.DataFrame({
        'Feature': categorical_features,
        'Chi2 Score': chi2_scores,
        'p-value': p_values
    })
    
    # Determine selected features based on the significance level
    selected_features = chi2_results[chi2_results['p-value'] < significance_level]['Feature'].tolist()
    not_selected_features = chi2_results[chi2_results['p-value'] >= significance_level]['Feature'].tolist()
    
    return chi2_results, selected_features, not_selected_features
def perform_anova_test(df, numerical_features, target_variable):
    # Perform ANOVA F-test
    anova_f_scores, anova_p_values = f_classif(df[numerical_features], df[target_variable])
    
    # Create a DataFrame for the results
    anova_results = pd.DataFrame({
        'Feature': numerical_features,
        'F-Score': anova_f_scores,
        'p-value': anova_p_values
    })
    
    # Determine selected features based on the p-value threshold
    selected_features = anova_results[anova_results['p-value'] < 0.01]['Feature'].tolist()
    not_selected_features = anova_results[anova_results['p-value'] >= 0.01]['Feature'].tolist()
    
    return anova_results, selected_features, not_selected_features
def analyze_target_and_correlations(df, target_variable):
    y = df[target_variable]
    n_classes = y.nunique()
    print(f"Number of unique classes in the target variable: {n_classes}")
    print("Number of values in each class:")
    print(y.value_counts())
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    correlation_with_target = correlation_matrix[target_variable].sort_values(ascending=False)
    print("Correlation with target variable:")
    print(correlation_with_target)
    
    # Identify features to remove based on correlation threshold
    high_correlation_features = correlation_with_target[(correlation_with_target > 0.9) | (correlation_with_target < -0.9)].index
    # high_correlation_features = high_correlation_features.drop(target_variable)  # Ensure we don't drop the target itself
    if len(high_correlation_features) > 0:
        print(f"Removing features with high correlation to target: {list(high_correlation_features)}")
    # df = df.to_numpy()
    df = df.drop(columns=high_correlation_features)
    return df, y
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error for {model.__class__.__name__}: {mse}")

df_cleaned = pd.read_csv('temp.csv', sep='|')
categorical_columns = ['gender_cv_id','legal_form_enum','has_email_address','interest_period_frequency_enum','interest_method_enum','interest_calculated_in_period_enum','term_frequency','number_of_repayments','transaction_type_enum']
date_columns = ['activation_date','office_joining_date','date_of_birth','approvedon_date','expected_disbursedon_date','disbursedon_date','expected_maturedon_date','maturedon_date','transaction_date','submitted_on_date']
numerical_columns = ['status_enum','principal_amount','nominal_interest_rate_per_period','annual_nominal_interest_rate','principal_repaid_derived','principal_outstanding_derived','interest_charged_derived','interest_repaid_derived','interest_outstanding_derived','total_repayment_derived','total_costofloan_derived','total_outstanding_derived','amount','principal_portion_derived','outstanding_loan_balance_derived']
other_columns_not_encoded = ['has_mobile_no','validatedon_userid','loan_transaction_strategy_id','is_reversed','submittedon_date_client','submittedon_date_loan','validatedon_date','created_date','principal_amount_proposed','principal_disbursed_derived','total_expected_repayment_derived','total_expected_costofloan_derived','manually_adjusted_or_reversed']
df_cleaned = convert_to_categorical(df_cleaned, categorical_columns)
df_cleaned = process_date_columns(df_cleaned, date_columns)
df_cleaned = df_cleaned.drop(columns=other_columns_not_encoded)

df_cleaned = remove_columns_with_missing_values(df_cleaned, missing_threshold=0)

highly_corr = ["total_costofloan_derived","total_repayment_derived","principal_repaid_derived"
]
df_cleaned = df_cleaned.drop(columns=highly_corr)

print(df_cleaned.shape)

target_variable = 'interest_repaid_derived'

print(df_cleaned.shape)

X,y = analyze_target_and_correlations(df_cleaned, target_variable)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
ridge = Ridge(alpha=0.1)

# List of models to train
models = [lr, dt, ridge]

# Train and evaluate each model
for model in models:
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
