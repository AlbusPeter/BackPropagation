Neural Network
---
Neural Networks have been developed for decades. In this project, we are
going to achieve a simple neural network, explore the updating rules for parameters,
i.e. the backpropagation and try to make the learning progress more efficient by
modifying certain sections in our network, such as the activation function and the
learning method. In the project, we firstly implement a simple Neural Network
which achieves a 0.93 accuracy on the test set. By making changes to optimize our
networks, we finally achieve a 0.975 accuracy on the test set.

Backpropagation Algorithm
---
For the detailed illustration, please go to [Michael Nielsen's Blog](http://neuralnetworksanddeeplearning.com/index.html).<br>
Different from his blog, we use the cross-entropy as the cost function and use the softmax as the output activation function. Moreover, according to the work of [LeCun, Yann, et al](https://link.springer.com/chapter/10.1007/3-540-49430-8_2), we optimized our network and finally obtain a 97.5% accuracy.


Requirement
---

Python 2.7


Usage
-----------
```sh
git clone https://github.com/AlbusPeter/BackPropagation.git
cd BackPropagation
python main.py
```

Parameters
----
Network Initialize:<br>
	>Network(sizes, gradientCheck = False, shuffle_status = False, activation_func = None, adaptInitial = False, momentumUpdate = False)<br>
	>Paremeters:<br>
		>>sizes: indicates the number of neurons in each layer (store the numbers in list)<br>
		>>gradientCheck: indicates whether to check the gradients when doing the backpropagation. The default value is False.<br>
		>>shuffle_status: indicates whether to shuffle the training data before training. The default value is False.<br>
		>>activation_func: indicates the activation function type. The default value is None which indicates the sigmoid function. Avalaible type {'relu','tanh'}.<br>
		>>adaptInitial: indicates whether to use a specific initialization value. The default value is False.<br>
		>>momentumUpdate: indicates whether to use momentum to update the network. The default value is False.<br>
	>Example:<br>
	>>net = Network([784,64,10], gradientCheck = False, shuffle_status = True, activation_func = 'relu', adaptInitial = True, momentumUpdate = True)<br>

Training:<br>
	>Mini_GD(training_data, epochs, mini_batch_size, learning_rate, test_data = None, T = 50, regularization = 'L2', reg_lam = 0.0001)<br>
	>Paremeters:<br>
		>>training_data: the data used for training<br>
		>>epochs: the maximum iteration number<br>
		>>mini_batch_size: the size for each mini_batch. For size=1, it becomes the [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) algorithm.<br>
		>>learning_rate: the rate of gradient descend<br>
		>>test_data: the data used for testing the network. The default is None.<br>
		>>T: the decaying coefficient for the learning_rate. The default value is 50.<br>
		>>regularization: indicates the regularization type. The default one is 'L2'. Avalaible type {'L1','L2',None}. None for no regularization.<br>
		>>reg_lam: the regularization coefficient<br>
	>Example:<br>
	>>net.Mini_GD(training_data, 50, 128, 0.01, test_data=testing_data)<br>


If you have any question, let me know: `AlbusPeter.rzh@gmail.com` .
		# BackPropagation
