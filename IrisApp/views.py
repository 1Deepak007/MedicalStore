from django.http import JsonResponse
import pandas as pd
from django.shortcuts import render
from .models import PredResults
from django.shortcuts import render, redirect, HttpResponse
from IrisApp.models import *

# =====================================
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# =====================================

# Create your views here.

def home(request):
    return render(request,'IrisApp/home.html')

def diabetese(request):
    return render(request,"IrisApp/diabetese.html")


def maleria(request):
    return render(request, "IrisApp/maleria.html")


# def analyze(request):
#     if (request.POST):
#         data = request.POST.dict()
#         sepal_length = data.get('sepal_length')
#         sepal_width = data.get('sepal_width')
#         petal_length = data.get('petal_length')
#         petal_width = data.get('petal_width')
        
#         print("Sepal length is : ",sepal_length)
#     return HttpResponse("Analyze function ran")    




def heartdiseaseanalyze(request):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    data = request.POST.dict()
    
    # receiving data from home.html
    if request.method == 'POST':
        age = request.POST.get('age')
        sex = request.POST.get('sex')
        cp = request.POST.get('cp')
        trestbps = request.POST.get('trestbps')
        chol = request.POST.get('chol')
        fbs = request.POST.get('fbs')
        restecg = request.POST.get('restecg')
        thalach = request.POST.get('thalach')
        exang = request.POST.get('exang')
        oldpeak = request.POST.get('oldpeak')
        slope = request.POST.get('slope')
        ca = request.POST.get('ca')
        thal = request.POST.get('thal')
        # target = request.POST.get('target')              # (do not input target 1==have disease, 0==not have disease)
            
        print(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)            
        #----> loading the csv data to a Pandas DataFrame
        heart_data = pd.read_csv('IrisApp\heart_disease_data.csv')
        #----> print first 5 rows of the dataset
        heart_data.head()
        print(heart_data.head)
        #----> number of rows and columns in the dataset
        heart_data.shape
        #----> getting some info about the data
        heart_data.info()
        #----> checking for missing values
        heart_data.isnull().sum()
        #----> statistical measures about the data
        heart_data.describe()
        #----> checking the distribution of Target Variable
        heart_data['target'].value_counts()
        X = heart_data.drop(columns='target', axis=1)
        Y = heart_data['target']
        print(X)
        print(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        print(X.shape, X_train.shape, X_test.shape)
        model = LogisticRegression()
        #----> training the LogisticRegression model with Training data
        model.fit(X_train, Y_train)
        #----> accuracy on training data
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        print('Accuracy on Training data : ', training_data_accuracy)
        #----> accuracy on test data
        X_test_prediction = model.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
        print('Accuracy on Test data : ', test_data_accuracy)
        
        
        # input_data = (62,   0,  0,  140,    268,    0,  0,  160,    0,  3.6,    0,  2,  2)
        input_data = float(age), float(sex),float(cp), float(trestbps),float(chol),float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)
        print(type(input_data))

        print("THE INPUT DATA IS : ")
        print(input_data)
        #----> change the input data to a numpy array
        input_data_as_numpy_array= np.asarray(input_data)

        #----> reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = model.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0]== 0):
            return HttpResponse('<h2 style="background-color : aquamarine; text-align: center; padding-top: 5px; padding-bottom:5px; ">You do not have a Heart Disease</h2>')
            # print('The Person does not have a Heart Disease')
        else:
            return HttpResponse('<h2 style="background-color : aquamarine; text-align: center; padding-top: 5px; padding-bottom:5px; ">You have Heart Disease</h2>')
            # print('The Person has Heart Disease')
            


def diabeteseanalyze(request):    
    # Importing libraries
    import pandas as pd 
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    
    if request.method == 'POST':
        glucose = request.POST.get('glucose')
        insuline = request.POST.get('insuline')
        bmi = request.POST.get('bmi')
        age = request.POST.get('age')
        
            
        # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
        
        warnings.filterwarnings('ignore')

        # Importing dataset
        dataset = pd.read_csv('IrisApp\diabetes.csv')
        # Preview data
        dataset.head()
        # Dataset dimensions - (rows, columns)
        dataset.shape
        # Features data-type
        dataset.info()
        # Statistical summary
        dataset.describe().T
        # Count of null values
        dataset.isnull().sum()
        # Outcome countplot
        sns.countplot(x = 'Outcome',data = dataset)
        # Histogram of each feature
        import itertools

        col = dataset.columns[:8]
        plt.subplots(figsize = (20, 15))
        length = len(col)

        for i, j in itertools.zip_longest(col, range(length)):
            plt.subplot((length/2), 3, j + 1)
            plt.subplots_adjust(wspace = 0.1,hspace = 0.5)
            dataset[i].hist(bins = 20)
            plt.title(i)
        plt.show()

        # Scatter plot matrix 
        from pandas.plotting import scatter_matrix
        # from pandas.tools.plotting import scatter_matrix
        scatter_matrix(dataset, figsize = (20, 20));

        # Pairplot 
        sns.pairplot(data = dataset, hue = 'Outcome')
        plt.show()

        dataset_new = dataset

        # Replacing zero values with NaN
        dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN) 

        # Count of NaN
        dataset_new.isnull().sum()

        # Replacing NaN with mean values
        dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
        dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
        dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
        dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
        dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)

        # Statistical summary
        dataset_new.describe().T

        # Feature scaling using MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        dataset_scaled = sc.fit_transform(dataset_new)

        dataset_scaled = pd.DataFrame(dataset_scaled)

        # Selecting features - [Glucose, Insulin, BMI, Age]
        X = dataset_scaled.iloc[:, [glucose, insuline, bmi, age]].values               # <===========================
        # X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values                                           
        Y = dataset_scaled.iloc[:, 8].values

        # Splitting X and Y
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

        # Checking dimensions
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("Y_train shape:", Y_train.shape)
        print("Y_test shape:", Y_test.shape)

        # Logistic Regression Algorithm   --> Supervised ML algo, uses sigmoid fxn, gives o/p in 0 or 1
        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression(random_state = 42)
        logreg.fit(X_train, Y_train)

        # Plotting a graph for n_neighbors 
        from sklearn import metrics
        from sklearn.neighbors import KNeighborsClassifier

        X_axis = list(range(1, 31))
        acc = pd.Series()
        x = range(1,31)

        for i in list(range(1, 31)):
            knn_model = KNeighborsClassifier(n_neighbors = i) 
            knn_model.fit(X_train, Y_train)
            prediction = knn_model.predict(X_test)
            acc = acc.append(pd.Series(metrics.accuracy_score(prediction, Y_test)))
        plt.plot(X_axis, acc)
        plt.xticks(x)
        plt.title("Finding best value for n_estimators")
        plt.xlabel("n_estimators")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.show()
        print('Highest value: ',acc.values.max())

        # K nearest neighbors Algorithm
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)
        knn.fit(X_train, Y_train)

        # Support Vector Classifier Algorithm
        from sklearn.svm import SVC
        svc = SVC(kernel = 'linear', random_state = 42)
        svc.fit(X_train, Y_train)

        # Naive Bayes Algorithm
        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
        nb.fit(X_train, Y_train)

        # Decision tree Algorithm
        from sklearn.tree import DecisionTreeClassifier
        dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
        dectree.fit(X_train, Y_train)

        # Random forest Algorithm
        from sklearn.ensemble import RandomForestClassifier
        ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
        ranfor.fit(X_train, Y_train)

        # Making predictions on test dataset
        Y_pred_logreg = logreg.predict(X_test)
        Y_pred_knn = knn.predict(X_test)
        Y_pred_svc = svc.predict(X_test)
        Y_pred_nb = nb.predict(X_test)
        Y_pred_dectree = dectree.predict(X_test)
        Y_pred_ranfor = ranfor.predict(X_test)
        # Step 5: Model Evaluation
        # Evaluating using accuracy_score metric
        from sklearn.metrics import accuracy_score
        accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
        accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
        accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
        accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
        accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
        accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)
        # Accuracy on test set
        print("Logistic Regression: " + str(accuracy_logreg * 100))
        print("K Nearest neighbors: " + str(accuracy_knn * 100))
        print("Support Vector Classifier: " + str(accuracy_svc * 100))
        print("Naive Bayes: " + str(accuracy_nb * 100))
        print("Decision tree: " + str(accuracy_dectree * 100))
        print("Random Forest: " + str(accuracy_ranfor * 100))
        #From the above comparison, we can observe that K Nearest neighbors gets the highest accuracy of 78.57 %
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(Y_test, Y_pred_knn)
        cm
        # Heatmap of Confusion matrix
        sns.heatmap(pd.DataFrame(cm), annot=True)
        # Classification report
        from sklearn.metrics import classification_report
        print("===============================================================================")
        print(classification_report(Y_test, Y_pred_knn))
        res = classification_report(Y_test, Y_pred_knn)
        return HttpResponse(res)


    

def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "IrisApp/result.html", data)  
        
    


def index(request):
    return render(request, 'IrisApp/predict.html')

def result(request):
    return render(request, 'IrisApp/result.html')


# def predict_changes(request):
    
#     # receiving data from predict.html
#     if request.POST.get('action') == 'post':
#         sepal_length = float(request.POST.get('sepal_length'))
#         sepal_width = float(request.POST.get('sepal_width'))
#         petal_length = float(request.POST.get('petal_length'))
#         petal_width = float(request.POST.get('petal_width'))
        
        
#         # analyzing data
#         model = pd.read_pickle(r"C:\Users\deepa\OneDrive\Desktop\Sem4 Project\MedicalStoreProject\IrisApp\static\new_model.pickle")
        
#         # Make prediction
#         result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        
#         classification = result[0]
        
#         PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width, classification=classification)
        
#         # sending data(result) to predect.html 
#         return JsonResponse({'result':classification, 'sepal_length':sepal_length,  'sepal_width':sepal_width, 'petal_length':petal_length, 'petal_width':petal_width}, safe=False)
        
#         # PredResults.objects.create(sepal_length = sepal_length, sepal_width = sepal_width, petal_length=petal_length, petal_width=petal_width, classification=classification)


# def view_results(request):
#     # Submit prediction and show all
#     data = {"dataset": PredResults.objects.all()}
#     return render(request, "results.html", data)
    
    
