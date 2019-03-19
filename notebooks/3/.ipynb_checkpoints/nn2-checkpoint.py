import numpy as np
from matplotlib import pyplot as plt

X_train = np.array(
   [[0, 0, 1], 
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]])
y_train = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(sig):
    return sig*(1-sig)

#Weights
w0 = 2*np.random.random((3, 5)) - 1 #for input   - 4 inputs, 3 outputs
w1 = 2*np.random.random((5, 1)) - 1 #for layer 1 - 5 inputs, 3 outputs

#learning rate
n = 0.1

#Errors - for graph later
errors = []

DEBUG = False
#Train
for i in range(10000):

    #Feed forward
    layer0 = X_train
    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1, w1))

    #Back propagation using gradient descent
    layer2_error =  2 * (y_train - layer2)
    layer2_delta = layer2_error * sigmoid_deriv(layer2) / 4

    error = np.mean((y_train - layer2) ** 2)

    if DEBUG:
        print(layer1.T.dot(layer2_delta))
        for i in range(5):
            d_w = np.zeros_like(w1)
            d_w[i] = 0.000001
            l0 = X_train
            l1 = sigmoid(np.dot(l0, w0))
            l2 = sigmoid(np.dot(l1, w1 + d_w))
            e = np.mean((y_train - l2) ** 2)
            print((error - e)/ 0.000001)


    layer1_error = layer2_delta.dot(w1.T)
    layer1_delta = layer1_error * sigmoid_deriv(layer1)

    if DEBUG:
        print(layer0.T.dot(layer1_delta))
        for i in range(5):
            for j in range(3):
                d_w = np.zeros_like(w0)
                d_w[j][i] = 0.0000001
                l0 = X_train
                l1 = sigmoid(np.dot(l0, w0 + d_w))
                l2 = sigmoid(np.dot(l1, w1))
                e = np.mean((y_train - l2) ** 2)
                print((error - e)/ 0.0000001)

    w1 += layer1.T.dot(layer2_delta) * n
    w0 += layer0.T.dot(layer1_delta) * n
    
    errors.append(error)
    accuracy = (1 - error) * 100

#Plot the accuracy chart
plt.plot(errors)
plt.xlabel('Training')
plt.ylabel('Error')
plt.show()
        
print("Training Accuracy " + str(round(accuracy,2)) + "%")
