from importos import *
from Network import *
import numpy as np


training_data,testing_data = Getdata()

# print('Start bii10')
# net = Network([784,10,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
# net.Mini_GD(training_data, 3600, 60000, 0.00001, test_data=testing_data, reg_lam = 0)
# net.plot_error()

# print('Start bii20')
# net = Network([784,20,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
# net.Mini_GD(training_data, 3600, 60000, 0.00001, test_data=testing_data, reg_lam = 0)
# net.plot_error()

print('Start bii50')
net = Network([784,50,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.00001, test_data=testing_data, reg_lam = 0)
net.plot_error('5bii50.png')

#####################################################
print('Start biii10sigmoid')
net = Network([784,10,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 500)
net.plot_error('biii10sigmoid.png')

print('Start biii10relu')
net = Network([784,10,10], gradientCheck = False, shuffle_status = False, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 500)
net.plot_error('biii10relu.png')

print('Start biii20sigmoid')
net = Network([784,20,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 500)
net.plot_error('biii20sigmoid.png')

print('Start biii20relu')
net = Network([784,20,10], gradientCheck = False, shuffle_status = False, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 500)
net.plot_error('biii20relu.png')

print('Start biii50sigmoid')
net = Network([784,50,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 500)
net.plot_error('biii50sigmoid.png')

print('Start biii50relu')
net = Network([784,50,10], gradientCheck = False, shuffle_status = False, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 500)
net.plot_error('biii50relu.png')
#####################################################

#####################################################
print('Start biv10sigmoid')
net = Network([784,10,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 50)
net.plot_error('biv10sigmoid.png')

print('Start biv10relu')
net = Network([784,10,10], gradientCheck = False, shuffle_status = False, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 50)
net.plot_error('biv10relu.png')

print('Start biv20sigmoid')
net = Network([784,20,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 50)
net.plot_error('biv20sigmoid.png')

print('Start biv20relu')
net = Network([784,20,10], gradientCheck = False, shuffle_status = False, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 50)
net.plot_error('biv20relu.png')

print('Start biv50sigmoid')
net = Network([784,50,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 50)
net.plot_error('biv50sigmoid.png')

print('Start biv50relu')
net = Network([784,50,10], gradientCheck = False, shuffle_status = False, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 3600, 60000, 0.000002, test_data=testing_data, reg_lam = 50)
net.plot_error('biv50relu.png')
#####################################################

#####################################################

print('Start cii')
net = Network([784,10], gradientCheck = False, shuffle_status = False, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 18, 1, 0.01, test_data=testing_data, reg_lam = 0)
net.plot_error('cii.png')

#####################################################

#####################################################
print('Start ciii10sigmoid')
net = Network([784,10,10], gradientCheck = False, shuffle_status = True, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 15, 1, 0.01, test_data=testing_data, reg_lam = 0)
net.plot_error('ciii10sigmoid.png')

print('Start ciii10relu')
net = Network([784,10,10], gradientCheck = False, shuffle_status = True, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 15, 1, 0.002, test_data=testing_data, reg_lam = 0)
net.plot_error('ciii10relu.png')

print('Start ciii20sigmoid')
net = Network([784,20,10], gradientCheck = False, shuffle_status = True, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 15, 1, 0.01, test_data=testing_data, reg_lam = 0)
net.plot_error('ciii20sigmoid.png')

print('Start ciii20relu')
net = Network([784,20,10], gradientCheck = False, shuffle_status = True, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 15, 1, 0.002, test_data=testing_data, reg_lam = 0)
net.plot_error('ciii20relu.png')

print('Start ciii50sigmoid')
net = Network([784,50,10], gradientCheck = False, shuffle_status = True, activation_func = 'sigmoid', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 15, 1, 0.01, test_data=testing_data, reg_lam = 0)
net.plot_error('ciii50sigmoid.png')

print('Start ciii50relu')
net = Network([784,50,10], gradientCheck = False, shuffle_status = True, activation_func = 'relu', adaptInitial = False, momentumUpdate = False)
net.Mini_GD(training_data, 15, 1, 0.002, test_data=testing_data, reg_lam = 0)
net.plot_error('ciii50relu.png')
#####################################################