import pandas as pd
import random
import math
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import warnings

warnings.filterwarnings('ignore')

data = {
    'cgpa':["y","n","y","n","y","y"],
    "interactiveness":["g","g","a","a","g","g"],
    "practical":["g","m","m","g","m","m"],
    "communication":["y","n","n","n","y","y"],
    'target': ["y","y","n","n","y","y"]}

def shuffle_within_columns(data):
    for col in data:
        random.shuffle(data[col])
    return data

data= shuffle_within_columns(data)
df = pd.DataFrame(data)

def Counterdata(array):
    return {list(set(array))[i]:i for i in range(len(set(array)))}

#based on decision stump
def AdaBoostAlgorithm(data,targetname):
    for i in data.columns.tolist():
        data[i]=[float(dict(Counterdata(list(data[i])))[j]) for j in data[i]]
    bootstrap_samples = data.sample(frac=1, replace=True)
    target=bootstrap_samples[targetname]
    weights=[1/len(target)]*len(target)
    bootstrap_samples.drop(["target"],axis=1, inplace=True)
    print(bootstrap_samples.columns.tolist())
    alphainstances={}
    for j in bootstrap_samples.columns.tolist():
        error=0
        alpha=0
        normalizing_factor=0
        correct_weights=[]
        incorrect_weights=[]
        dict1={}
        print(len(target))
        clf = tree.DecisionTreeClassifier(max_depth=1,).fit(bootstrap_samples,target,sample_weight=weights)
        prediction=clf.predict(bootstrap_samples)
        for i in range(len(target)):
            # print(error)
            if prediction[i]==list(target)[i]:
                error=error+weights[i]
                correct_weights.append(weights[i])
                dict1[i]=1
            else:
                incorrect_weights.append(weights[i])
                dict1[i]=0
        if error>1:
            alphainstances[j]="useless"
            continue
        alpha=math.log((1-error)/error)*0.5
        normalizing_factor=sum(correct_weights)*math.exp(-alpha)+sum(incorrect_weights)*math.exp(alpha)
        print(normalizing_factor,alpha,error,correct_weights,incorrect_weights)
        for i in dict1:
            if dict1[i]==1:
                weights[i]*=math.exp(-alpha)/normalizing_factor
            else:
                weights[i]*=math.exp(alpha)/normalizing_factor
        alphainstances[j]=alpha
    return alphainstances
print(AdaBoostAlgorithm(df,'target'))


        


    
    