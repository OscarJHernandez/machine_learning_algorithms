import numpy as np

# A class that implements a one layer neural network
class NeuralNet_OneLayer:
    
    
    def __init__(self,inputs,outputs,nodes):
        self.nodes = nodes # The number of nodes that will be generated
        self.inputs = inputs
        self.outputs = outputs
        
        # The input and output weight matrices
        self.Wi = np.random.rand(nodes,inputs+1) # The one is added as a bias
        self.Wf = np.random.rand(outputs,nodes)
        
        print("Wi paramenters: ", nodes*(inputs+1))
        print("Wf paramenters: ", outputs*nodes)
    
    # This defines the activation function
    def activation(self,x):
        y = 1.0/(1.0+np.exp(-x))
        return y
     
    # This carries out the forward pass through the network
    def forward_pass(self,x):
        self.x =np.append(x,[1.0]) # Add a one for the bias term
        self.z1 = np.matmul(self.Wi, self.x)
        self.v1 = self.activation(self.z1)
        self.z2 = np.matmul(self.Wf,self.v1)
        return self.z2
    
    # This reads in the training data
    def read_in_data(self,x_train,y_train):
        self.x_data = x_train
        self.y_data = y_train
    
    # This function computes the Loss function
    def loss_function(self):
        L = 0.0
        
        for i in range(0,len(self.x_data)):
            x_veci = self.x_data[i]
            y_veci = self.forward_pass(x_veci)
            yi = self.y_data[i]
            L = L+0.5*np.dot(yi-y_veci,yi-y_veci)
        
        
        return L
        
    
    
