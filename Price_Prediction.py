import pandas as pd
import chardet
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import Binarizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import chardet
#from sknn.mlp import Regressor, Layer
#import sknn.mlp

def clean_data(df):
    print(df.isnull().sum())
    #Drop commas from numbers with thousands so they can be converted to ints
    df = df.replace(',','', regex=True)
    #Fill NaN values with mean from the column
    df['Travel time (public transport) to a GP premises \n(minutes)'].fillna((df['Travel time (public transport) to a GP premises \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a dentist \n(minutes)'].fillna((df['Travel time (public transport) to a dentist \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a pharmacist \n(minutes)'].fillna((df['Travel time (public transport) to a pharmacist \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to an optician \n(minutes)'].fillna((df['Travel time (public transport) to an optician \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a Job Centre or Jobs and Benefits Office (minutes)'].fillna((df['Travel time (public transport) to a Job Centre or Jobs and Benefits Office (minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a Post Office \n(minutes)'].fillna((df['Travel time (public transport) to a Post Office \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a supermarket / food store \n(minutes)'].fillna((df['Travel time (public transport) to a supermarket / food store \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a large service centre (minutes)'].fillna((df['Travel time (public transport) to a large service centre (minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a library \n(minutes)'].fillna((df['Travel time (public transport) to a library \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to a Council Leisure Centre or sports facilities \n(minutes)'].fillna((df['Travel time (public transport) to a Council Leisure Centre or sports facilities \n(minutes)'].mean()), inplace=True)
    df['Travel time (public transport) to financial services (minutes)'].fillna((df['Travel time (public transport) to financial services (minutes)'].mean()), inplace=True)
    df.drop(columns= ['Travel time (public transport) to a day nursery or crŠche (minutes)', 'Travel time (public transport) to a restaurant  \n(minutes)', 'Travel time (public transport) to a \nfastfood outlet\n(minutes)', 'Travel time (public transport) to a pub (minutes)', 'Travel time (public transport) to a health & beauty\nestablishment (minutes)'], axis = 1, inplace=True)
    #Convert strings representing categories to ints
    df['SA'] = pd.factorize(df['SA'])[0].astype(np.uint16)
    df['SA Code'] = pd.factorize(df['SA Code'])[0].astype(np.uint16)
    return df
                                                        
                                                                                        
                                                           

def test_classification(model,X_train, X_test, y_train, y_test):
    print(X_train.shape)
    print(y_train.shape)
    print("Test Model accuracy: ")
    print(model.score(X_test, y_test))
    test_predictions = model.predict(X_test)
    #TP/TP+FN
    #Answers question: Of all cases of churn how many did we identify
    micro_recall = metrics.recall_score(y_test, test_predictions)
    print("Micro averaged recall score: {0:0.4f}".format(micro_recall) )
    #Precision = True positives/All positives
    #High precision indicates a low false positive rate
    micro_precision = metrics.precision_score(y_test, test_predictions)
    print("Micro averaged precision score: {0:0.4f}".format(micro_precision) )

#Evaluates how effective a model is at predicting the outcome
def test_model(regressor, X_train, X_test, y_train, y_test):
    #print(y_train.describe())
# =============================================================================
#     acc_score = regressor.score(X_test, y_test)
#     print("R^2 score")
#     print(acc_score)
#     y_pred = regressor.predict(X_test)
#     print("Mean absolute Error: ")
#     print(mean_absolute_error(y_pred, y_test))
#     print("Mean Square Error: ")
#     print(mean_squared_error(y_pred, y_test))
#     print(" ")
# =============================================================================
    y_train_pred = regressor.predict(X_train)
    print("Train Mean absolute Error: ")
    print(mean_absolute_error(y_train, y_train_pred))
    print("Train Mean Square Error: ")
    print(mean_squared_error(y_train, y_train_pred))
    acc_score = regressor.score(X_test, y_test)
    print("R^2 score")
    print(acc_score)
    y_pred = regressor.predict(X_test)
    print("Mean absolute Error: ")
    print(mean_absolute_error(y_test, y_pred))
    print("Mean Square Error: ")
    print(mean_squared_error(y_test, y_pred))
    print(" ")
    
#Converts problem from regression to classification so that different svm regularization techniques can be used
def convert_to_classification(X_train, X_test, y_train, y_test, threshold, num_features):
    #Discretize y values i.e. convert the target variable from a continuous value into categorical
    #Median target variable value used as threshold
    transformer = Binarizer(threshold)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    X_test = X_test.reshape(-1, num_features)
    X_train = X_train.reshape(-1, num_features)
    y_train_discretized = transformer.fit_transform(y_train)
    y_test_discretized = transformer.fit_transform(y_test)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train_discretized.shape)
    print(y_test_discretized.shape)
    return X_train, X_test, y_train_discretized, y_test_discretized

def support_vector_machine(X_train, y_train):
    regressor = SVR(kernel='linear', C=50, max_iter = 5000, verbose=True)
    regressor.fit(X_train, y_train)
    print("SVR Coefficients")
    print(regressor.coef_)
    
    return regressor
    
def lin_regression(X_train, y_train):
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def ridge_regression(X_train, y_train):
    ridge = linear_model.Ridge()
    ridge.fit(X_train, y_train)
    return ridge

def lasso_regression(X_train, y_train):
    lasso = linear_model.Lasso()
    lasso.fit(X_train, y_train)
    return lasso

def elastic_net_regression(X_train, y_train):
    en = linear_model.ElasticNet()
    en.fit(X_train, y_train)
    return en

def svc_l1(X_train, y_train):
    lin_svc = LinearSVC(penalty = 'l1', dual=False, max_iter=7000)
    lin_svc.fit(X_train, y_train)
    return lin_svc

def svc_l2(X_train, y_train):
    lin_svc = LinearSVC(penalty = 'l2', max_iter=7000)
    lin_svc.fit(X_train, y_train)
    return lin_svc

def mlp_l1(X_train, y_train):
    
    return 0

def mlp_reg(X_train, y_train):
    mlp = MLPRegressor(random_state=1, max_iter=500, alpha = 0.1)
    mlp.fit(X_train, y_train)
    return mlp

def boston_data_analysis():
    X, y = load_boston(return_X_y=True)
    print(type(y))
    print(np.median(y))
    X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.2)
    print("SVR")
    regressor = support_vector_machine(X_train, y_train)
    test_model(regressor,X_train, X_test, y_train, y_test )
    print("Linear Regression")
    lr = lin_regression(X_train, y_train)
    test_model(lr,X_train, X_test, y_train, y_test )
    print("Ridge Regression")
    ridge = ridge_regression(X_train, y_train)
    test_model(ridge,X_train, X_test, y_train, y_test )
    print("Lasso Regression")
    lasso = ridge_regression(X_train, y_train)
    test_model(lasso,X_train, X_test, y_train, y_test )
    print("Elastic Net Regression")
    en = elastic_net_regression(X_train, y_train)
    test_model(en,X_train, X_test, y_train, y_test )
    
# =============================================================================
#     #Convert the regression problem to a classification problem so that SVM can be done with l1 and L2 regularization
#     X_train, X_test, y_train_classification, y_test_classification = convert_to_classification(X_train, X_test, y_train, y_test, 21.2, 13)
#     
#     #Create and test a Support Vector Classifier with L1 regularization
#     svc = svc_l1(X_train, y_train_classification)
#     test_classification(svc,X_train, X_test, y_train_classification, y_test_classification )
#     
#     #Create and test a Support Vector Classifier with L1 regularization
#     svc_2 = svc_l2(X_train, y_train_classification)
#     test_classification(svc_2,X_train, X_test, y_train_classification, y_test_classification )
# =============================================================================
    
# =============================================================================
#     mlp = mlp_reg(X_train, y_train)
#     test_model(mlp,X_train, X_test, y_train, y_test )
# =============================================================================

def main():
    boston_data_analysis()
    
    
if __name__ == "__main__":
    main()
