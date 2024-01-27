import random
import math

def RNN(input_seq,output_seq,epochs,learning_rate):
    weightsx=[random.uniform(-0.5, 0.5) for i in range(len(input_seq))]
    weightsh=[random.uniform(-0.5, 0.5) for i in range(len(input_seq))]
    weightsy=[random.uniform(-0.5, 0.5) for i in range(len(input_seq))]
    biash=[random.uniform(-0.5, 0.5) for i in range(len(input_seq))]
    biasy=[random.uniform(-0.5, 0.5) for i in range(len(input_seq))]
    for i in range(epochs):
        result=[]
        h=0
        valuesofh=[]
        #forward propagation
        for j in range(len(input_seq)):
            h=math.tanh(weightsx[j]*input_seq[j]+weightsh[j]*h+biash[j])
            valuesofh.append(h)
            y=weightsy[j]*h+biasy[j]
            if y>=0.2:
                result.append(1)
            else:
                result.append(0)
        #cross-entropy
        correct_classification=0
        print(result)
        for j in range(len(input_seq)):
            if output_seq[j]==result[j]:
                correct_classification+=1
        loss=-math.log(correct_classification)
        #optimization
        for j in range(len(weightsx)):
            weightsx[j]+=loss*learning_rate*input_seq[j]
        for j in range(len(weightsh)):
            weightsh[j]+=loss*learning_rate*valuesofh[j]
        for j in range(len(weightsy)):
            weightsy[j]+=loss*learning_rate*output_seq[j]
        for j in range(len(biash)):
            biash[j]+=loss*learning_rate
        for j in range(len(biasy)):
            biasy[j]+=loss*learning_rate
    print(loss)
RNN([1,0,1,1,0],[0,0,1,1,1],10,0.4)
            
                

