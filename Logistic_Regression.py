import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def sigmodial_function(z):
    return 1/(1+ np.exp(-z))

def net_input_function(weights, x_data):
    return np.dot(x_data, weights[1:]) + weights[0]

def predict(weights, x_data):
    z= net_input_function(weights, x_data)
    return np.where(sigmodial_function(z)>=0.5, 1, 0)

def Weights_upgradation(weights, x_data, y_pred, y_actual, Learning_rate=0.05):
    error= y_actual- y_pred
    weights[1:]= weights[1:]+ Learning_rate*(np.dot(x_data.T, error))
    weights[0]= weights[0]+ np.sum(error)
    
    return weights
    
def Logistic_Regression(x_data, y_data, no_of_epochs=25, Learning_rate=0.05):
    rows, attributes = x_data.shape
    weights = np.zeros((attributes+1))
    
    costs=[]
    
    for i in range(no_of_epochs):
        z= net_input_function(weights, x_data)
        y_predict= sigmodial_function(z)
        
        Weights_upgradation(weights, x_data, y_predict, y_data, Learning_rate)
        x=0.01
        y_predict+= np.where(y_predict<0.01, x, 0)
        y_predict-= np.where((1-y_predict)<0.01, x, 0)
        
        
        cost=(-(y_data).dot(np.log(y_predict))-(1-y_data).dot(np.log(1-y_predict)))/len(x_data)
        costs.append(cost)
    
    x= np.arange(no_of_epochs)
    plt.title("Logistic Regression")
    plt.xlabel("Epoch No")
    plt.ylabel("Cost Function Value")
    plt.plot(x, costs, color='red', marker='*')
    plt.show()
    return  weights

def accuracy(actual, predict):
    count=0
    for i in range(len(actual)):
        if actual[i]==predict[i]:
            count= count+1
    return count/len(actual)


iris= load_iris()
print(iris.DESCR)

x_data= iris.data
print("\n\nThe x_dataset: \n",x_data)
y_data= iris.target
print("\n\nThe y_dataset: \n",y_data)

class_labels= np.unique(y_data)
print("The class labels are: ", class_labels)

y_data_class1= np.zeros(len(y_data))
for i in range(len(y_data)):
    if(y_data[i]==class_labels[0]):
        y_data_class1[i] = 1


x_train, x_test, y_train, y_test= train_test_split(x_data, y_data_class1, test_size=0.4, shuffle=True)
weights= Logistic_Regression(x_train, y_train, no_of_epochs=10)

print("\nThe weights are: \n", weights)

y_predicted= predict(weights, x_test)
print("\nThe accuarcy of the predicted results is: ", accuracy(y_test, y_predicted))