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
    'target': ['Yes', 'No', 'Yes', 'No', 'Yes','Neutral','No','Neutral','Yes','No']
}

def shuffle_within_columns(data):
    for col in data:
        random.shuffle(data[col])
    return data

data= shuffle_within_columns(data)
df = pd.DataFrame(data)

def RandomForestClassifier(data,targetname,bootstrap=5,max_depth=2):
    for i in data.columns.tolist():
        data[i]=[float(dict(Counter(list(data[i])))[j]) for j in data[i]]
    random_row=data.sample().index[0]
    dropped_row=data.loc[random_row].copy()
    target_of_dropped_row=dropped_row[targetname]
    data.drop(index=random_row,inplace=True)
    dropped_row.drop([targetname], inplace=True)
    bootstrap_samples = [data.sample(frac=1, replace=True) for _ in range(bootstrap)]
    answer=[]
    for i in bootstrap_samples:
        target=i[targetname]
        i.drop([targetname],axis=1,inplace=True)
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
        clf = clf.fit(i,target)
        y_pred = clf.predict([dropped_row])
        answer.append(y_pred[0])
    return max(set(answer), key=answer.count)

print(RandomForestClassifier(df,'target'))


    
        



    