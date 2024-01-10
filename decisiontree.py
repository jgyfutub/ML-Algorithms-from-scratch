import pandas as pd
from collections import Counter
import math
import random

def shuffle_within_columns(data):
    for col in data:
        random.shuffle(data[col])
    return data

##for only categorical data
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
    "Song": ["Bohemian Rhapsody", "Bohemian Rhapsody", "Imagine", "Imagine", "Hallelujah", "Hallelujah", "Yesterday", "Yesterday", "Happy", "Happy"]
}
target={
    'target': ['Yes', 'No', 'Yes', 'No', 'Yes','Neutral','No','Neutral','Yes','No']
}
data= shuffle_within_columns(data)
df = pd.DataFrame(data)
target=pd.DataFrame(target)
def DecisionTreeC45(data,target):
    if len(target['target'])==1:
        return 
    unique=Counter(target['target'])
    entropytarget=0
    for i in unique.keys():
        entropytarget=entropytarget+(unique[i]/len(target['target']))*math.log2(unique[i]/len(target['target']))
    dictdata={}
    for i in data.select_dtypes(include='object').columns:
        dictcolumns={}
        for j in list(set(data[i])):
            dictcolumns.update({j:{}})
        for j in range(len(data[i])):
            if target['target'][j] not in dictcolumns[data[i][j]].keys():
                dictcolumns[data[i][j]].update({target['target'][j]:1})
            else:
                dictcolumns[data[i][j]][target['target'][j]]+=1
        columnentropy=0
        for j in dictcolumns:
            sumof=sum(dictcolumns[j].values())
            entropytargetrow=0
            for k in dictcolumns[j]:
                entropytargetrow=entropytargetrow+(dictcolumns[j][k]/sumof)*math.log2(dictcolumns[j][k]/sumof)
            columnentropy=columnentropy+(sumof/len(data[i]))*(-1)*entropytargetrow
        dictdata.update({i:-entropytarget-columnentropy})
    print(dictdata)
    if dictdata=={}:
        return
    else:
        max_key = max(dictdata, key=dictdata.get)
        newdatas={}
        newtargets={}
        for i in set(data[max_key]):
            newdatas[i]={}
            newtargets[i]={'target': []}
            for j in data.select_dtypes(include='object').columns:
                if j!=max_key:
                    newdatas[i][j]=[]
        for i in range(len(data[max_key])):
            for j in newdatas[data[max_key][i]]:
                newdatas[data[max_key][i]][j].append(data[j][i])
            newtargets[data[max_key][i]]['target'].append(target['target'][i])
        print(max_key)
        for i in newdatas:
            DecisionTreeC45(pd.DataFrame(newdatas[i]),pd.DataFrame(newtargets[i]))
        
    return 

    
print(DecisionTreeC45(df,target))

