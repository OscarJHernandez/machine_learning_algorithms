import numpy as np
import matplotlib.pyplot as plt
import neural_net as nn



nn1 = nn.NeuralNet_OneLayer(inputs=2,outputs=2,nodes=1)

x = 3.0*np.ones(2)

print(nn1.forward_pass(x))


x1 =  np.asarray([0.0,2.0])
x2 =  np.asarray([1.0,1.0])
x3 =  np.asarray([-1.0,-1.0])
x_train = np.append([x1],[x2],axis=0)
x_train = np.append(x_train,[x3],axis=0)
y_train = x_train

nn1.read_in_data(x_train,y_train)
print(nn1.loss_function())
