import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def Eucledian_Distance(p1, p2):
    dist=0
    dist= np.sum((p1-p2)**2)
        
    return np.sqrt(dist)

def KNN_Classifier(x_data, y_data, x_test, k):
    predicted_label=[]
    for item in range(len(x_test)):
        points_dist= []
        for i in range(len(x_data)):
            dist= Eucledian_Distance(x_data[i, :], item)
            points_dist.append(dist)
        
        points_dist= np.array(points_dist)
        
        minimum_k_distances= np.argsort(points_dist)[:k]
        class_labels= y_data[minimum_k_distances]
        label= statistics.mode(class_labels)
        predicted_label.append(label)
        
    return predicted_label

def accuracy(actual, predict):
    count=0
    for i in range(len(actual)):
        if actual[i]==predict[i]:
            count= count+1
    return count/len(actual)

iris= load_iris()
print(iris.DESCR)


x= iris.data
print("\n\nThe x_dataset: \n",x)
y= iris.target
print("\n\nThe y_dataset: \n",y)

print("\n",x.shape)
output_labels= np.unique(y)
print("The different class labels present in Iris Dataset are: \n",output_labels)
       
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.1, stratify=y, shuffle=True, random_state=85)

k=5
predicted_label= np.array(KNN_Classifier(x_train, y_train, x_test, k))
actual_label= np.array(y_test)

print("\n\nActual Labels of test data are: \n", actual_label)
print("\n\nPredicted Labels of test data are: \n", predicted_label)
print("\nThe accuarcy of the predicted results is: ", accuracy(actual_label, predicted_label))

