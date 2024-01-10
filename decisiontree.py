import pandas as pd
from collections import Counter
import math
##for only categorical data
data = {
    'column1': ['A', 'B', 'A', 'C', 'B' ,'B'],
    'column2': ['Red', 'Green', 'Blue', 'Green', 'Red','Green'],
    'column4': ['Small', 'Medium', 'Large', 'Small', 'Medium', 'Large'],
    'column5': ['North', 'South', 'East', 'West', 'South','East'],
}
target={
    'target': ['Yes', 'No', 'Yes', 'No', 'Yes','Neutral']
}
df = pd.DataFrame(data)
target=pd.DataFrame(target)
def DecisionTreeC45(data,target):
    unique=Counter(target['target'])
    entropytarget=0
    for i in unique.keys():
        entropytarget=entropytarget+(unique[i]/len(target))*math.log2(unique[i]/len(target))
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
    max_key = max(dictdata, key=dictdata.get)
    print(data.loc[1])
    # newdata={}
    # newdatatarget={}
    # for i in set(data[max_key]):
    #     newdata[i]={}
    #     newdatatarget[i]={}
    # for i in len(data[max_key]):
        
        
        
    return max_key

    
print(DecisionTreeC45(df,target))

