#!/usr/bin/env python
# coding: utf-8

# In[619]:
# In[620]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


def plot():
    # In[621]:
    data=pd.read_csv("StudentProgress.csv")
    data.head()

    # In[622]:
    # convert final_score to categorical variable # Good:15~20 Fair:10~14 Poor:0~9
    data['final_grade'] = 'Na'
    data.loc[(data.Final >= 15) & (data.Final <= 20), 'final_grade'] = 'good'
    data.loc[(data.Final >= 10) & (data.Final <= 14), 'final_grade'] = 'fair'
    data.loc[(data.Final >= 0) & (data.Final <= 9), 'final_grade'] = 'poor'
    data.head(5)

    # In[623]:
    data.isnull().sum()

    # In[624]:
    # Final Grade Countplot
    plt.figure(figsize=(8,6))
    # sns.countplot(data.final_grade, order=["poor","fair","good"], palette='Set1')
    sns.countplot(data=data, x='final_grade', order=["poor", "fair", "good"], palette='Set1')
    plt.title('Final Grade - Number of Students',fontsize=20)
    plt.xlabel('Final Grade', fontsize=16)
    plt.ylabel('Number of Student', fontsize=16)

    # In[625]:
    le =  LabelEncoder()

    le.fit(data.Gender.drop_duplicates())
    data.Gender = le.transform(data.Gender)
    print(data.Gender)

    le.fit(data.address.drop_duplicates())
    data.address = le.transform(data.address)
    print(data.address)

    le.fit(data.Pstatus.drop_duplicates())
    data.Pstatus = le.transform(data.Pstatus)
    print(data.Pstatus)

    le.fit(data.Mjob.drop_duplicates())
    data.Mjob = le.transform(data.Mjob)
    print(data.Mjob)

    le.fit(data.Fjob.drop_duplicates())
    data.Fjob = le.transform(data.Fjob)
    print(data.Fjob)

    le.fit(data.guardian.drop_duplicates())
    data.guardian = le.transform(data.guardian)
    print(data.guardian)

    le.fit(data.schoolsup.drop_duplicates())
    data.schoolsup = le.transform(data.schoolsup)
    print(data.schoolsup)

    le.fit(data.famsup.drop_duplicates())
    data.famsup = le.transform(data.famsup)
    print(data.famsup)

    le.fit(data.activities.drop_duplicates())
    data.activities = le.transform(data.activities)
    print(data.activities)

    le.fit(data.higher.drop_duplicates())
    data.higher = le.transform(data.higher)
    print(data.higher)

    le.fit(data.internet.drop_duplicates())
    data.internet = le.transform(data.internet)
    print(data.internet)

    le.fit(data.romantic.drop_duplicates())
    data.romantic = le.transform(data.romantic)
    print(data.romantic)

    le.fit(data.final_grade.drop_duplicates())
    data.final_grade = le.transform(data.final_grade)
    print(data.final_grade)

    # In[626]:
    df = data.drop(columns=['school', 'famsize', 'reason', 'paid', 'nursery'], axis=1)
    df.head()

    # In[627]:
    numeric_data= df.select_dtypes(include=[np.number])
    categorical_data = df.select_dtypes(exclude=[np.number])
    print ("There are {} numeric and {} categorical columns in dataset"
    .format(numeric_data.shape[1],categorical_data.shape[1]))

    # In[628]:
    # see correlation between variables through a correlation heatmap
    corr = df.corr()
    plt.figure(figsize=(20,15))
    sns.heatmap(corr, annot=True, cmap="Reds")
    plt.title('Correlation Heatmap', fontsize=20)

    # In[629]:
    # romantic status
    percent = (lambda x: x/x.sum())
    index = [0, 1, 2]
    romance = pd.crosstab(index=data.final_grade, columns=data.romantic)
    romance_tab = np.log(romance)
    romance_perc = romance_tab.apply(percent).reindex(index)
    plt.figure()
    romance_perc.plot.bar(colormap="PiYG_r", fontsize=16, figsize=(8,8))
    plt.title('Final Grade By Romantic Status', fontsize=20)
    plt.ylabel('Percentage of Logarithm Student Counts ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.show()

    # In[630]:
    # chi-square test result -- significant!
    romance_table = sm.stats.Table(romance)
    romance_rslt = romance_table.test_nominal_association()
    romance_rslt.pvalue

    # In[631]:
    # weekend alcohol consumption
    alc_tab1 = pd.crosstab(index=data.final_grade, columns=data.Walc)
    alc_tab = np.log(alc_tab1)
    alc_perc = alc_tab.apply(percent).reindex(index)

    # create good student dataframe
    good = data.loc[data.final_grade == 2]
    good['good_alcohol_usage']=good.Walc
    # create fair student dataframe
    fair = data.loc[data.final_grade == 1]
    fair['fair_alcohol_usage']=fair.Walc
    # create poor student dataframe
    poor = data.loc[data.final_grade == 0]
    poor['poor_alcohol_usage']=poor.Walc

    plt.figure(figsize=(10,6))
    p1=sns.kdeplot(good['good_alcohol_usage'], shade=True, color="r")
    p1=sns.kdeplot(fair['fair_alcohol_usage'], shade=True, color="g")
    p1=sns.kdeplot(poor['poor_alcohol_usage'], shade=True, color="b")
    plt.title('Good Performance vs. Fair Performance vs. Poor Performance Student Weekend Alcohol Consumption', fontsize=20)
    plt.ylabel('Density', fontsize=16)
    plt.xlabel('Level of Alcohol Consumption', fontsize=16)

    # In[632]:
    alc_perc.plot.bar(colormap="Reds", figsize=(10,8), fontsize=16)
    plt.title('Final Grade By Weekend Alcohol Consumption', fontsize=20)
    plt.ylabel('Percentage of Logarithm Student Counts', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)

    # In[633]:
    # chi-square test result -- significant!
    alc_table = sm.stats.Table(alc_tab1)
    alc_rslt = alc_table.test_nominal_association()
    alc_rslt.pvalue

    # In[634]:
    good['good_student_father_education'] = good.Fedu
    fair['fair_student_father_education'] = fair.Fedu
    poor['poor_student_father_education'] = poor.Fedu
    good['good_student_mother_education'] = good.Medu
    fair['fair_student_mother_education'] = fair.Medu
    poor['poor_student_mother_education'] = poor.Medu

    # In[635]:
    # see the difference between good and poor performers' father education level(numeric: from 1 - very low to 5 - very high)
    plt.figure(figsize=(6,4))
    p2=sns.kdeplot(good['good_student_father_education'], shade=True, color="r")
    p2=sns.kdeplot(fair['fair_student_father_education'], shade=True, color="g")
    p2=sns.kdeplot(poor['poor_student_father_education'], shade=True, color="b")
    plt.xlabel('Father Education Level', fontsize=20)

    # In[636]:
    # see the difference between good and poor performers' mother education level(numeric: from 1 - very low to 5 - very high)
    plt.figure(figsize=(6,4))
    p3=sns.kdeplot(good['good_student_mother_education'], shade=True, color="r")
    p3=sns.kdeplot(fair['fair_student_mother_education'], shade=True, color="g")
    p3=sns.kdeplot(poor['poor_student_mother_education'], shade=True, color="b")
    plt.xlabel('Mother Education Level', fontsize=20)

    # In[637]:
    # use OLS to see coefficients
    X_edu = data[['Medu','Fedu']]
    y_edu = data.Final
    edu = sm.OLS(y_edu, X_edu)
    results_edu = edu.fit()
    results_edu.summary()

    # In[638]:
    # going out with friends (numeric: from 1 - very low to 5 - very high)
    plt.figure(figsize=(6,10))
    sns.boxplot(x='goingOut', y='Final', data=data, palette='hot')
    plt.title('Final Grade By Frequency of Going Out', fontsize=20)
    plt.ylabel('Final Score', fontsize=16)
    plt.xlabel('Frequency of Going Out', fontsize=16)

    # In[639]:
    out_tab = pd.crosstab(index=data.final_grade, columns=data.goingOut)
    out_perc = out_tab.apply(percent).reindex(index)
    out_perc.plot.bar(colormap="mako_r", fontsize=16, figsize=(14,6))
    plt.title('Final Grade By Frequency of Going Out', fontsize=20)
    plt.ylabel('Percentage of Student', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)

    # In[640]:
    # chi-square test result -- significant!
    out_table = sm.stats.Table(out_tab)
    out_rslt = out_table.test_nominal_association()
    out_rslt.pvalue

    # In[641]:
    # Desire for higher education and study time by age
    plt.figure(figsize=(12,8))
    sns.violinplot(x='age', y='studytime', hue='higher', data=data, palette="Accent_r", ylim=(1,6))
    plt.title('Distribution Of Study Time By Age & Desire To Receive Higher Education', fontsize=20)
    plt.ylabel('Study Time', fontsize=16)
    plt.xlabel('Age', fontsize=16)

    # In[642]:
    plt.figure(figsize=(12,8))
    sns.barplot(data=data, x='age', y='studytime', hue='higher')
    plt.title('Distribution Of Study Time By Age & Desire To Receive Higher Education', fontsize=20)
    plt.ylabel('Study Time', fontsize=16)
    plt.xlabel('Age', fontsize=16)

    # In[643]:
    higher_tab = pd.crosstab(index=data.final_grade, columns=data.higher)
    higher_perc = higher_tab.apply(percent).reindex(index)
    higher_perc.plot.bar(colormap="tab20_r", figsize=(14,6), fontsize=16)
    plt.title('Final Grade By Desire to Receive Higher Education', fontsize=20)
    plt.xlabel('Final Grade', fontsize=16)
    plt.ylabel('Percentage of Student', fontsize=16)

    # In[644]:
    # chi-square test result -- significant!
    higher_table = sm.stats.Table(higher_tab)
    higher_rslt = higher_table.test_nominal_association()
    higher_rslt.pvalue

    # In[645]:
    # living area: urban vs. rural
    plt.rc("axes",labelsize=13)
    plt.rc("xtick",labelsize=13)
    plt.rc("ytick",labelsize=13)
    sns.countplot(x=data["address"]).set_title("Urban and Rural Students Count")

    # In[646]:
    percent = lambda x: x / x.sum()
    index = [0, 1, 2]
    tab1 = pd.crosstab(index=data['final_grade'], columns=data['address'])
    tab = np.log(tab1 + 1)  # Apply logarithm after adding 1 to handle zeros
    perc = tab.apply(percent).reindex(index)
    plt.figure()
    perc.plot.bar(colormap="PiYG_r", fontsize=16, figsize=(8, 8))
    plt.title('Final Grade By Living Area', fontsize=20)
    plt.ylabel('Percentage of Logarithm Student', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.show()

    # In[647]:
    # chi-square test result -- significant!
    ad_table = sm.stats.Table(tab1)
    ad_rslt = ad_table.test_nominal_association()
    ad_rslt.pvalue



    # In[648]:
    train = df.iloc[:, :]
    test = df.iloc[:, :]

    # In[649]:
    train.head()

    # In[650]:
    test.head()

    # In[651]:
    X = train.drop(columns=['Pstatus', 'Mjob', 'Fjob', 'Dalc', 'final_grade'], axis=1)
    y = train['final_grade']
    X.head()

    # In[652]:
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)

    # In[653]:
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # prediction of model
    y_pred = lr.predict(x_test)

    # training accuracy of model
    lr.score(x_train, y_train)

    # test accuracy of model
    lr.score(x_test, y_test)

    # creating a function to create adhusted R-Squared
    def adj_r2(X, y, model):
        r2 = model.score(X, y)
        n = X.shape[0]
        p = X.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return adjusted_r2

    print(adj_r2(x_train, y_train, lr))
    print(adj_r2(x_test, y_test, lr))



    from sklearn.linear_model import Lasso, LassoCV
    lasso_cv = LassoCV(alphas = None, cv = 10, max_iter = 100000)
    lasso_cv.fit(x_train, y_train)

    # best alpha parameter
    alpha = lasso_cv.alpha_
    alpha

    lasso = Lasso(alpha = lasso_cv.alpha_)
    lasso.fit(x_train, y_train)

    lasso.score(x_train, y_train)
    lasso.score(x_test, y_test)

    print(adj_r2(x_train, y_train, lasso))
    print(adj_r2(x_test, y_test, lasso))


    # In[654]:
    from sklearn.linear_model import Ridge, RidgeCV
    alphas = np.random.uniform(0, 10, 50)
    ridge_cv = RidgeCV(alphas = alphas, cv = 10)
    ridge_cv.fit(x_train, y_train)

    # best alpha parameter
    alpha = ridge_cv.alpha_
    alpha

    ridge = Ridge(alpha = ridge_cv.alpha_)
    ridge.fit(x_train, y_train)

    ridge.score(x_train, y_train)
    ridge.score(x_test, y_test)
    print(adj_r2(x_train, y_train, ridge))
    print(adj_r2(x_test, y_test, ridge))


    # In[655]:
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    elastic_net_cv = ElasticNetCV(alphas = None, cv = 10, max_iter = 100000)
    elastic_net_cv.fit(x_train, y_train)

    # best alpha parameter
    alpha = elastic_net_cv.alpha_
    alpha

    # l1 ratio
    elastic_net_cv.l1_ratio

    elastic_net = ElasticNet(alpha = elastic_net_cv.alpha_, l1_ratio = elastic_net_cv.l1_ratio)
    elastic_net.fit(x_train, y_train)

    elastic_net.score(x_train, y_train)
    elastic_net.score(x_test, y_test)

    print(adj_r2(x_train, y_train, elastic_net))
    print(adj_r2(x_test, y_test, elastic_net))


    # In[656]:
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train, y_train)


    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.model_selection import cross_val_score
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))
    accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 5)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


    # In[657]:
    from sklearn.model_selection import train_test_split, cross_val_score
    # classify column
    def classify(model):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model.fit(x_train, y_train)
        print('Accuracy:', model.score(x_test, y_test))

        score = cross_val_score(model, X, y, cv=5)
        print('CV Score:', np.mean(score))

        accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 5)
        print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


    # In[658]:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    classify(model)


    # In[659]:
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    classify(model)


    # In[660]:
    from xgboost import XGBClassifier
    model = XGBClassifier()
    classify(model)


    # In[661]:
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(verbose=0)
    classify(model)


    # In[662]:
    model = RandomForestClassifier()
    model.fit(X, y)
    test.head()


    # In[663]:
    X_test = test.drop(columns=['Pstatus', 'Mjob', 'Fjob', 'Dalc', 'final_grade'], axis=1)
    X_test.head()


    # In[664]:
    pred = model.predict(X_test)
    print(pred)


    # In[665]:
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    accuracy = accuracy_score(y_test,y_pred)
    precision =  precision_score(y_test,y_pred,average="weighted")
    recall = recall_score(y_test,y_pred,average="weighted")
    f1 = f1_score(y_test,y_pred,average="weighted")
    print("Accuracy - {}".format(accuracy))
    print("Precision - {}".format(precision))
    print("Recall- {}".format(recall))
    print("f1 - {}".format(f1))


    # In[666]:
    dfd = data.copy()
    dfd = dfd.drop([ 'Final'], axis=1)


    # In[667]:
    # label encode final_grade
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    dfd.final_grade = le.fit_transform(dfd.final_grade)


    # In[668]:
    # dataset train_test_split
    from sklearn.model_selection import train_test_split
    X = dfd.drop('final_grade',axis=1)
    y = dfd.final_grade
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


    # In[669]:


    # get dummy varibles
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # see total number of features
    len(list(X_train))


    # In[670]:
    # find the optimal # of minimum samples leaf
    from sklearn.tree import DecisionTreeClassifier
    msl=[]
    for i in range(1,39):
        tree = DecisionTreeClassifier(min_samples_leaf=i)
        t= tree.fit(X_train, y_train)
        ts=t.score(X_test, y_test)
        msl.append(ts)
    msl = pd.Series(msl)
    msl.where(msl==msl.max()).dropna()


    # In[671]:
    # final model
    tree = DecisionTreeClassifier(min_samples_leaf=17)
    t= tree.fit(X_train, y_train)
    print("Decisioin Tree Model Score" , ":" , t.score(X_train, y_train) , "," ,
        "Cross Validation Score" ,":" , t.score(X_test, y_test))


    # In[672]:
    # find a good # of estimators
    from sklearn.ensemble import RandomForestClassifier

    ne=[]
    for i in range(1,39):
        forest = RandomForestClassifier()
        f = forest.fit(X_train, y_train)
        fs = f.score(X_test, y_test)
        ne.append(fs)
    ne = pd.Series(ne)
    ne.where(ne==ne.max()).dropna()


    # In[673]:


    # find a good # of min_samples_leaf
    from sklearn.ensemble import RandomForestClassifier

    ne=[]
    for i in range(1,39):
        forest = RandomForestClassifier(n_estimators=36, min_samples_leaf=i)
        f = forest.fit(X_train, y_train)
        fs = f.score(X_test, y_test)
        ne.append(fs)
    ne = pd.Series(ne)
    ne.where(ne==ne.max()).dropna()


    # In[674]:
    # final model
    forest = RandomForestClassifier(n_estimators=36, min_samples_leaf=2)
    f = forest.fit(X_train, y_train)
    print("Raondom Forest Model Score" , ":" , f.score(X_train, y_train) , "," ,
        "Cross Validation Score" ,":" , f.score(X_test, y_test))


    # In[675]:
    from sklearn.svm import SVC
    svc = SVC()
    s= svc.fit(X_train, y_train)
    print("SVC Model Score" , ":" , s.score(X_train, y_train) , "," ,
        "Cross Validation Score" ,":" , s.score(X_test, y_test))


    # In[676]:


    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',fit_intercept=True)


    # find optimal # of features to use in the model
    from sklearn.feature_selection import SelectKBest, chi2

    ks=[]
    for i in range(1,39):
        sk = SelectKBest(chi2, k=i)
        x_new = sk.fit_transform(X_train,y_train)
        x_new_test=sk.fit_transform(X_test,y_test)
        l = lr.fit(x_new, y_train)
        ll = l.score(x_new_test, y_test)
        ks.append(ll)

    ks = pd.Series(ks)
    ks = ks.reindex(list(range(1,39)))
    ks

    plt.figure(figsize=(10,5))
    ks.plot.line()
    plt.title('Feature Selction', fontsize=20)
    plt.xlabel('Number of Feature Used', fontsize=16)
    plt.ylabel('Prediction Accuracy', fontsize=16)


    # In[677]:
    ks.where(ks==ks.max()).dropna()


    # In[678]:
    # final model
    sk = SelectKBest(chi2, k=8)
    x_new = sk.fit_transform(X_train,y_train)
    x_new_test=sk.fit_transform(X_test,y_test)
    lr = lr.fit(x_new, y_train)
    print("Logistic Regression Model Score" , ":" , lr.score(x_new, y_train) , "," ,
        "Cross Validation Score" ,":" , lr.score(x_new_test, y_test))


    # In[679]:
    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier(n_estimators=2)
    af = ada.fit(X_train, y_train)
    print("Ada Boost Model Score" , ":" , af.score(X_train, y_train) , "," ,
        "Cross Validation Score" ,":" , af.score(X_test, y_test))


    # In[680]:
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier()
    sf = sgd.fit(X_train, y_train)
    print("Stochastic Gradient Descent Model Score" , ":" , sf.score(X_train, y_train) , "," ,
        "Cross Validation Score" ,":" , sf.score(X_test, y_test))

    return data