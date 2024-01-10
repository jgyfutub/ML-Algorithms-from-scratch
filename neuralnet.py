import random
import math
def MLP(arr,output,hiddenneurons,epochs,learning_rate):
    weights=[random.uniform(-0.5, 0.5) for i in range(len(arr)*hiddenneurons)]
    bias=[random.uniform(-0.5, 0.5) for i in range(hiddenneurons)]
    outputbias=random.uniform(-0.5, 0.5)
    hiddenweights=[random.uniform(-0.5, 0.5) for i in range(hiddenneurons)]
    for i in range(epochs):
        #forward propagation
        hiddenneuronsresult=[1/(1+math.exp(-1*(sum([x * y for x, y in zip(arr, weights[len(arr)*i:len(arr)*(i+1)])])+bias[i]))) for i in range(hiddenneurons)]
        outputnet=1/(1+math.exp(-1*(sum([x * y for x, y in zip(hiddenweights, hiddenneuronsresult)])+outputbias)))
        print(outputnet)
        #backward propagation
        errorinoutputlayer=outputnet*(1-outputnet)*(output-outputnet)
        hiddenlayererror=[hiddenneuronsresult[i]*(1-hiddenneuronsresult[i])*errorinoutputlayer*hiddenweights[i] for i in range(hiddenneurons)]
        for i in range(len(weights)):
            weights[i]+=learning_rate*hiddenlayererror[i//len(arr)]*arr[i%len(arr)]
        for i in range(len(hiddenweights)):
            hiddenweights[i]+=learning_rate*hiddenneuronsresult[i]*errorinoutputlayer
        for i in range(len(bias)):
            bias[i]+=learning_rate*hiddenlayererror[i]
        outputbias+=learning_rate*errorinoutputlayer
        
        # print(hiddenneuronsresult,hiddenlayererror)
    print(outputnet)
MLP([1,0,1,1,0,1,0,0,1],1,6,10,0.4)