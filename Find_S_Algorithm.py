import pandas as pd
import numpy as np
 
dataset = pd.read_csv("Find_S_Dataset.csv")
print(dataset)
print("\n")

x_data = np.array(dataset)[:,:-1]
print("The x_data: \n",x_data)
print("\n")

y_data = np.array(dataset)[:,-1]
print("The no of class labels are:\n", np.unique(y_data))
print("\n")

def train(x,y):
    for idx, output in enumerate(y):
        if output == "Yes":
            specific_hypothesis = x[idx].copy()
            break
    
    for idx, val in enumerate(x):
        if y[idx] == "Yes":
            for a in range(len(specific_hypothesis)):
                if val[a] != specific_hypothesis[a]:
                    specific_hypothesis[a] = '?'
                else:
                    pass
            print(specific_hypothesis)
                 
    return specific_hypothesis

#obtaining the final hypothesis
print("\n\nThe final hypothesis is:",train(x_data,y_data))