import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def analyze_student_progress(csv_file):
    # Load the dataset
    data = pd.read_csv(csv_file)

    # Convert final_score to categorical variable
    data['final_grade'] = 'Na'
    data.loc[(data.Final >= 15) & (data.Final <= 20), 'final_grade'] = 'good'
    data.loc[(data.Final >= 10) & (data.Final <= 14), 'final_grade'] = 'fair'
    data.loc[(data.Final >= 0) & (data.Final <= 9), 'final_grade'] = 'poor'

    # Label encode categorical columns
    le = preprocessing.LabelEncoder()
    columns_to_encode = ['Gender', 'address', 'Pstatus', 'Mjob', 'Fjob', 'guardian', 'schoolsup', 'famsup',
                         'activities', 'higher', 'internet', 'romantic', 'final_grade']
    for column in columns_to_encode:
        data[column] = le.fit_transform(data[column])

    # Drop unnecessary columns
    df = data.drop(columns=['school', 'famsize',
                   'reason', 'paid', 'nursery'], axis=1)

    # Split the dataset into numeric and categorical data
    # numeric_data = df.select_dtypes(include=[np.number])
    # categorical_data = df.select_dtypes(exclude=[np.number])

    # Generate correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=True, cmap="Reds")
    plt.title('Correlation Heatmap', fontsize=20)
    plt.savefig('heatmap.png')

    # Train-test split
    X = df.drop(columns=['Pstatus', 'Mjob', 'Fjob',
                'Dalc', 'final_grade'], axis=1)
    y = df['final_grade']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Decision Tree Model
    dt_model = DecisionTreeClassifier(min_samples_leaf=17)
    dt_model.fit(X_train, y_train)
    dt_train_score = dt_model.score(X_train, y_train)
    dt_test_score = dt_model.score(X_test, y_test)

    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=36, min_samples_leaf=2)
    rf_model.fit(X_train, y_train)
    rf_train_score = rf_model.score(X_train, y_train)
    rf_test_score = rf_model.score(X_test, y_test)

    # Support Vector Machine (SVM) Model
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_train_score = svm_model.score(X_train, y_train)
    svm_test_score = svm_model.score(X_test, y_test)

    # Logistic Regression Model
    lr_model = LogisticRegression(
        multi_class='multinomial', solver='newton-cg', fit_intercept=True)
    lr_model.fit(X_train, y_train)
    lr_train_score = lr_model.score(X_train, y_train)
    lr_test_score = lr_model.score(X_test, y_test)

    # Feature Selection with Logistic Regression
    skb = SelectKBest(chi2, k=8)
    x_new = skb.fit_transform(X_train, y_train)
    x_new_test = skb.fit_transform(X_test, y_test)
    lr_model_fs = LogisticRegression(
        multi_class='multinomial', solver='newton-cg', fit_intercept=True)
    lr_model_fs.fit(x_new, y_train)
    lr_train_score_fs = lr_model_fs.score(x_new, y_train)
    lr_test_score_fs = lr_model_fs.score(x_new_test, y_test)

    # AdaBoost Classifier
    ada_model = AdaBoostClassifier(n_estimators=2)
    ada_model.fit(X_train, y_train)
    ada_train_score = ada_model.score(X_train, y_train)
    ada_test_score = ada_model.score(X_test, y_test)

    # Stochastic Gradient Descent (SGD) Classifier
    sgd_model = SGDClassifier()
    sgd_model.fit(X_train, y_train)
    sgd_train_score = sgd_model.score(X_train, y_train)
    sgd_test_score = sgd_model.score(X_test, y_test)

    # Make predictions on the test set
    dt_predictions = dt_model.predict(X_test)
    rf_predictions = rf_model.predict(X_test)
    svm_predictions = svm_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)
    lr_predictions_fs = lr_model_fs.predict(x_new_test)
    ada_predictions = ada_model.predict(X_test)
    sgd_predictions = sgd_model.predict(X_test)

    # Calculate evaluation metrics
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_accuracy_fs = accuracy_score(y_test, lr_predictions_fs)
    ada_accuracy = accuracy_score(y_test, ada_predictions)
    sgd_accuracy = accuracy_score(y_test, sgd_predictions)

    dt_precision = precision_score(y_test, dt_predictions, average='weighted')
    rf_precision = precision_score(y_test, rf_predictions, average='weighted')
    svm_precision = precision_score(
        y_test, svm_predictions, average='weighted')
    lr_precision = precision_score(y_test, lr_predictions, average='weighted')
    lr_precision_fs = precision_score(
        y_test, lr_predictions_fs, average='weighted')
    ada_precision = precision_score(
        y_test, ada_predictions, average='weighted')
    sgd_precision = precision_score(
        y_test, sgd_predictions, average='weighted')

    dt_recall = recall_score(y_test, dt_predictions, average='weighted')
    rf_recall = recall_score(y_test, rf_predictions, average='weighted')
    svm_recall = recall_score(y_test, svm_predictions, average='weighted')
    lr_recall = recall_score(y_test, lr_predictions, average='weighted')
    lr_recall_fs = recall_score(y_test, lr_predictions_fs, average='weighted')
    ada_recall = recall_score(y_test, ada_predictions, average='weighted')
    sgd_recall = recall_score(y_test, sgd_predictions, average='weighted')

    dt_f1 = f1_score(y_test, dt_predictions, average='weighted')
    rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
    svm_f1 = f1_score(y_test, svm_predictions, average='weighted')
    lr_f1 = f1_score(y_test, lr_predictions, average='weighted')
    lr_f1_fs = f1_score(y_test, lr_predictions_fs, average='weighted')
    ada_f1 = f1_score(y_test, ada_predictions, average='weighted')
    sgd_f1 = f1_score(y_test, sgd_predictions, average='weighted')

    return {
        'dt_train_score': dt_train_score,
        'dt_test_score': dt_test_score,
        'rf_train_score': rf_train_score,
        'rf_test_score': rf_test_score,
        'svm_train_score': svm_train_score,
        'svm_test_score': svm_test_score,
        'lr_train_score': lr_train_score,
        'lr_test_score': lr_test_score,
        'lr_train_score_fs': lr_train_score_fs,
        'lr_test_score_fs': lr_test_score_fs,
        'ada_train_score': ada_train_score,
        'ada_test_score': ada_test_score,
        'sgd_train_score': sgd_train_score,
        'sgd_test_score': sgd_test_score,
        'dt_accuracy': dt_accuracy,
        'rf_accuracy': rf_accuracy,
        'svm_accuracy': svm_accuracy,
        'lr_accuracy': lr_accuracy,
        'lr_accuracy_fs': lr_accuracy_fs,
        'ada_accuracy': ada_accuracy,
        'sgd_accuracy': sgd_accuracy,
        'dt_precision': dt_precision,
        'rf_precision': rf_precision,
        'svm_precision': svm_precision,
        'lr_precision': lr_precision,
        'lr_precision_fs': lr_precision_fs,
        'ada_precision': ada_precision,
        'sgd_precision': sgd_precision,
        'dt_recall': dt_recall,
        'rf_recall': rf_recall,
        'svm_recall': svm_recall,
        'lr_recall': lr_recall,
        'lr_recall_fs': lr_recall_fs,
        'ada_recall': ada_recall,
        'sgd_recall': sgd_recall,
        'dt_f1': dt_f1,
        'rf_f1': rf_f1,
        'svm_f1': svm_f1,
        'lr_f1': lr_f1,
        'lr_f1_fs': lr_f1_fs,
        'ada_f1': ada_f1,
        'sgd_f1': sgd_f1
    }
