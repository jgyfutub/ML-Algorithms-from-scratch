import numpy as np
import pandas as pd
import random
import math

data = {
    'cgpa':np.random.rand(10),
    "interactiveness":np.random.rand(10),
    "practical":np.random.rand(10),
    "communication":np.random.rand(10),
    }

testdata = {
    'cgpa':np.random.rand(6),
    "interactiveness":np.random.rand(6),
    "practical":np.random.rand(6),
    "communication":np.random.rand(6),
    }
target={'target': np.random.rand(10)}
testtarget={'target': np.random.rand(6)}
df = pd.DataFrame(data)
target1=pd.DataFrame(target)
testdata1= pd.DataFrame(testdata)
testtarget=pd.DataFrame(testtarget)

def LinearRegression(data,target,testdata):
    numpydata=data.to_numpy()
    targetdata=target.to_numpy()
    transposeddata=np.transpose(numpydata)
    testnumpydata=testdata.to_numpy()
    result1=np.dot(np.dot(np.linalg.inv(np.dot(transposeddata,numpydata)),transposeddata),targetdata)
    result=np.dot(testnumpydata,result1)
    return result

result=LinearRegression(df,target1,testdata1)
print(result)
actual_values=testtarget.to_numpy()
mse = np.mean((actual_values - result) ** 2)
print("Mean Squared Error (MSE):", mse)