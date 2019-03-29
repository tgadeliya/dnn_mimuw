import numpy as np
class  softmax_regression:
    
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_tr = X
        self.y_tr = y
        self.w = np.zeros([X.shape[1], 10])
    
    def set_weights(self, weight):
        """
        Set matrix of weight
        """
        self.w = weight
    
    def get_weights():
        return self.w
    
    def softmax(self,z):
        z -= np.max(z)
        exp = np.exp(z)
        return exp/np.sum(exp)
    
    def predict(self, X):
        """
        Predict classes with softmax function and for  return  
        vector with probabilities of classes for every sample
        from X.
        
        Arguments: 
        w (n_features, n_class) - weights matrix (np.array)
        X (n_samples,n_features) - input matrix with samples(np.array)
        
        Returns:
        prediction (n_samples,n_class) - prediction of every class 
        for input samples
        """
        M = np.dot(X,self.w)
        return np.apply_along_axis(self.softmax, axis=1, arr=M)
    
    def predict_classes(self,X):
        return self.predict(X).argmax(axis=1)
    
    def cmt_loss_and_gradients(self, X, y, l2_reg):
        n_samples = X.shape[0]
        
        pred = self.predict(X)
        y_arr  = y.argmax(axis=1)
        #probabilities predicted by model for truth classes
        prob = pred[range(n_samples),y_arr]
        
        #Loss function with L2 regularisation
        loss = -(np.sum(np.log(prob))) + l2_reg * np.sum(self.w **2)
        
        #Gradient analytical computation
        grad = pred.copy()
        grad[range(n_samples),y_arr] -= 1
        grad = X.T.dot(grad/n_samples)
        #grad += l2_reg*2*w - gradient of regularization
        return loss,grad
    
    
    def train(self,lr=0.05, l2_reg=0.5, k_folds = 5, epochs = 3,batch_size=128): 
        """
        Perform training with pre-defined hyperparameters
        
        """
        n_iter = self.X_tr.shape[0] // batch_size
        self.losses=[]
        
        for i in range(epochs):
            for i in range(n_iter):
                p = i * batch_size
                q = (i+1) * batch_size 
                loss, grad = self.cmt_loss_and_gradients(self.X_tr, self.y_tr, l2_reg)
                self.losses.append(loss)
                self.w -= lr * grad