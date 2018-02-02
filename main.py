from importos import *
from Network import *
import numpy as np


training_data,testing_data = Getdata()

net = Network([784,64,10], gradientCheck = False, shuffle_status = False, activation_func = 'relu', adaptInitial = True, momentumUpdate = True)
net.Mini_GD(training_data, 50, 128, 0.01, test_data=testing_data)

# plot3path(net.train_acc_path[:-1],'Training Set',net.hold_acc_path[:-1],'Validate Set',net.test_acc_path,'Testing Set',[0.9,1], 'Iteration','Accuracy','Accuracy VS Iteration with Relu')
# plot3path(net.train_loss_path[:-1],'Training Set',net.hold_loss_path[:-1],'Validate Set',net.test_loss_path,'Testing Set',False, 'Iteration','Loss','Loss VS Iteration with Relu')