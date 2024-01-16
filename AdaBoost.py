import pandas as pd
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import warnings

warnings.filterwarnings('ignore')

data = {
    "Animal": ["Dog", "Dog", "Bird", "Fish", "Snake", "Tiger", "Tiger", "Elephant", "Monkey", "Bear"],
    "Fruit": ["Apple", "Apple", "Apple", "Grapefruit", "Strawberry", "Strawberry", "Mango", "Mango", "Kiwi", "Melon"],
    "Color": ["Red", "Red", "Red", "Yellow", "Blue", "Blue", "Pink", "Brown", "Brown", "Brown"],
    "City": ["London", "London", "Tokyo", "New York", "Berlin", "Rome", "Rome", "Beijing", "Moscow", "Sydney"],
    "Country": ["USA", "USA", "Mexico", "Brazil", "Argentina", "Chile", "Peru", "Colombia", "Venezuela", "Ecuador"],
    "Job": ["Doctor", "Doctor", "Teacher", "Teacher", "Engineer", "Engineer", "Musician", "Musician", "Athlete", "Athlete"],
    "Hobby": ["Reading", "Reading", "Writing", "Writing", "Movies", "Movies", "Games", "Games", "Camping", "Camping"],
    "Movie": ["Action", "Action", "Comedy", "Comedy", "Romance", "Romance", "Thriller", "Thriller", "Horror", "Horror"],
    "Book": ["The Lord of the Rings", "The Lord of the Rings", "Harry Potter", "Harry Potter", "To Kill a Mockingbird", "To Kill a Mockingbird", "1984", "1984", "Animal Farm", "Animal Farm"],
    "Song": ["Bohemian Rhapsody", "Bohemian Rhapsody", "Imagine", "Imagine", "Hallelujah", "Hallelujah", "Yesterday", "Yesterday", "Happy", "Happy"],
    'target': ['Yes', 'No', 'Yes', 'No', 'Yes','No','No','Yes','No','No']
}

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
    weights=[1/len(data[targetname])]*len(data[targetname])
    bootstrap_samples = data.sample(frac=1, replace=True)
    target=bootstrap_samples[targetname]
    bootstrap_samples.drop(["target"],axis=1, inplace=True)
    print(data['target'])
    for i in bootstrap_samples.columns.tolist():
        clf = tree.DecisionTreeClassifier(max_depth=1,).fit(bootstrap_samples,target,sample_weight=weights)
        prediction=clf.predict([bootstrap_samples[i]])
        print(prediction)
print(AdaBoostAlgorithm(df,'target'))


        


    
    