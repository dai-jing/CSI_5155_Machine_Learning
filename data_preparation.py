import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV, RFE
from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE


# Data Cleaning
def data_cleaning(X_df: pd.DataFrame, y_df: pd.DataFrame):
    X = X_df.copy()
    y = y_df.copy()

    # Step 1: Remove columns that contain a single value
    columns_single_value = [index for index, counts in enumerate(X.nunique()) if counts == 1]
    columns_single_value_display = pd.DataFrame({"features contain single value": X.columns[columns_single_value]})
    display(columns_single_value_display)
    X.drop(X.columns[columns_single_value], axis=1, inplace=True)

    # Step 2: Remove rows contain duplicate data
    if not X.duplicated().any():
        print("No duplicated rows")

    # Step 3: Handle missing data
    missing_values_columns = list()
    missing_values_counts = list()
    missing_values_percentages = list()
    columns_need_dropped = list()
    missing_value_columns = X.loc[:, X.isnull().any()]
    for missing_value_column in missing_value_columns:
        values_list = missing_value_columns[missing_value_column]
        missing_values_count = values_list.isna().sum()
        missing_values_percentage = missing_values_count / len(values_list) * 100
        if missing_values_percentage > 50:
            columns_need_dropped.append(missing_value_column)

        missing_values_columns.append(missing_value_column)
        missing_values_counts.append(missing_values_count)
        missing_values_percentages.append("{:0.2f}%".format(missing_values_percentage))

    display_df_dict = {"features contain missing values": missing_values_columns, "count": missing_values_counts, "percentage": missing_values_percentages}
    display(pd.DataFrame(display_df_dict))

    # Step 3.1: Drop features that has a missing values percentage > 50%
    X.drop(X[columns_need_dropped], axis=1, inplace=True)

    # Step 3.2: Fill missing values in medical_specialty feature with a new Missing value
    X["medical_specialty"].fillna("Missing", inplace=True)

    # Step 3.3: Imputation with mode value
    imputer = SimpleImputer(strategy="most_frequent")
    imputer.fit(X)
    X = pd.DataFrame(imputer.transform(X), columns=X.columns).astype(X.dtypes.to_dict())

    return X, y


def feature_encoding(X: pd.DataFrame, y: pd.DataFrame, age_as_target=False):
    if not age_as_target:
        # Handle feature 'age'
        X["age"] = X["age"].map({
            "[0-10)": 1,
            "[10-20)": 2,
            "[20-30)": 3,
            "[30-40)": 4,
            "[40-50)": 5,
            "[50-60)": 6,
            "[60-70)": 7,
            "[70-80)": 8,
            "[80-90)": 9,
            "[90-100)": 10
        }).astype("int64")

    # Handle feature 'diag'
    X["diag_1"] = X["diag_1"].map(lambda diag: group_name_from_icd9_code(diag))
    X["diag_2"] = X["diag_2"].map(lambda diag: group_name_from_icd9_code(diag))
    X["diag_3"] = X["diag_3"].map(lambda diag: group_name_from_icd9_code(diag))

    # Handle feature 'medical_specialty'
    X["medical_specialty"] = X["medical_specialty"].map(lambda medical_specialty: specialty_cluster_from_medical_specialty(medical_specialty))

    # Handle feature
    medication_changes_columns = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
                                  "acetohexamide",
                                  "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
                                  "miglitol", "troglitazone", "tolazamide", "insulin", "glyburide-metformin",
                                  "glipizide-metformin",
                                  "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
    for column in medication_changes_columns:
        X[column] = X[column].map(lambda medication_change: 0 if (medication_change == 'No' or medication_change == 'Steady') else 1).astype("category")

    X.drop(["payer_code"], axis=1, inplace=True)

    for column in X.columns:
        num_of_uniques = X[column].nunique()
        if num_of_uniques == 1:
            # single value features
            X.drop([column], axis=1, inplace=True)
        elif not X[column].dtypes == "int64":
            if num_of_uniques < 10:
                # categorical features
                X = pd.concat([X, pd.get_dummies(X[column], prefix=column).astype("int64")], axis=1)
                X.drop([column], axis=1, inplace=True)
            else:
                print("column: ", column)

    return X, y


# Feature Selection
def feature_selection(X: pd.DataFrame, y: pd.DataFrame):
    fe_column_transformer = ColumnTransformer(transformers=[
        ('numeric', SelectKBest(score_func=f_classif, k="all"), make_column_selector(dtype_include=np.number)),
        ('categorical', SelectKBest(score_func=chi2, k="all"), make_column_selector(dtype_include="category"))
    ])

    fe_column_transformer.fit(X, y)
    X = fe_column_transformer.transform(X)

    return X, y


# Feature Balancing
def data_balancing(X_df: pd.DataFrame, y_df: pd.DataFrame):
    sampling_model = SMOTE(random_state=1)
    X_oversampling, y_oversampling = sampling_model.fit_resample(X_df, y_df)

    print("after oversampling: \n", y_oversampling.value_counts(normalize=True))

    return X_oversampling, y_oversampling


def group_name_from_icd9_code(icd9_code):
    # clustering
    try:
        icd9_code_int = float(icd9_code)
        if 390 <= icd9_code_int <= 459 or icd9_code_int == 785:
            return "Circulatory"
        elif 460 <= icd9_code_int <= 519 or icd9_code_int == 786:
            return "Respiratory"
        elif 520 <= icd9_code_int <= 579 or icd9_code_int == 787:
            return "Digestive"
        elif icd9_code == "250.xx":
            return "Diabetes"
        elif 800 <= icd9_code_int <= 999:
            return "Injury"
        elif 710 <= icd9_code_int <= 739:
            return "Musculoskeletal"
        elif 580 <= icd9_code_int <= 629 or icd9_code_int == 788:
            return "Genitourinary"
        elif 140 <= icd9_code_int <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        return "Other"


def specialty_cluster_from_medical_specialty(medical_specialty):
    if "Surgery" in medical_specialty:
        return "Surgery"
    elif "Pediatrics" in medical_specialty:
        return "Pediatrics"
    elif "InternalMedicine" in medical_specialty:
        return "Internal Medicine"
    elif "Cardiology" in medical_specialty:
        return "Cardiology"
    elif "Family/GeneralPractice" in medical_specialty:
        return "Family/GeneralPractice"
    elif "Emergency/Trauma" in medical_specialty:
        return "Emergency/Trauma"
    elif "Missing" in medical_specialty:
        return "Missing"
    else:
        return "Other"


def get_feature_names(classifier: SelectKBest, names):
    mask = classifier.get_support()  # list of booleans
    new_features = []  # The list of your K best features

    for bool, feature in zip(mask, names):
        if bool:
            new_features.append(feature)

    return new_features
