import numpy as np 
import pandas as pd

dataset = pd.read_csv("Candidate_Elimination_Dataset.csv")
x_data = np.array(dataset.iloc[:,:-1])
print("\nX_Data Values :\n",x_data)

y_data = np.array(dataset.iloc[:,-1])
print("\ny_data Values : ",y_data)

def train(x_data, y_data): 
    specific_h = ['$' for i in range(len(x_data[0]))]
    print("\nInitialization of specific_h and genearal_h")
    print("\nSpecific Boundary: ", specific_h)
    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("\nGeneric Boundary: ",general_h)  

    for i, h in enumerate(x_data):
        print("\nInstance", i+1 , "is ", h)
        if y_data[i] == "yes":
            print("Instance is Positive ")
            for x in range(len(specific_h)):
                if specific_h[x]=='$':
                    specific_h[x]=h[x]
                    
                elif h[x]!= specific_h[x]:                    
                    specific_h[x] ='?'                     
                    general_h[x][x] ='?'
                   
        if y_data[i] == "no":            
            print("Instance is Negative ")
            for x in range(len(specific_h)): 
                if h[x]!= specific_h[x]:                    
                    general_h[x][x] = specific_h[x]                
                else:                    
                    general_h[x][x] = '?'        
        
        print("Specific Bundary after ", i+1, "Instance is ", specific_h)         
        print("Generic Boundary after ", i+1, "Instance is ", general_h)
        print("\n")

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]    
    for i in indices:   
        general_h.remove(['?', '?', '?', '?', '?', '?']) 
    return specific_h, general_h 

specific_final, gernal_final = train(x_data, y_data)

print("Final Specific_h: ", specific_final, sep="\n")
print("\n")
print("Final General_h: ", gernal_final, sep="\n")