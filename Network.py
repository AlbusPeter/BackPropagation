import random
import numpy as np
import math
import matplotlib.pyplot as plt

class Network(object):
	def __init__(self, sizes, gradientCheck = False, shuffle_status = False, activation_func = None, adaptInitial = False, momentumUpdate = False):
		self.numLayers = len(sizes)
		self.sizes = sizes
		self.gradientCheck = gradientCheck
		self.shuffle_status = shuffle_status
		self.activation_func = activation_func
		self.adaptInitial = adaptInitial
		self.momentumUpdate = momentumUpdate
		self.regularization = None
		self.reg_lam = 0.0001
		self.mu = 0.9
		self.v_momentum_b = [np.zeros((j,1)) for j in sizes[1:]]
		self.v_momentum_w = [np.zeros((i,j)) for i, j in zip(sizes[1:], sizes[:-1])]

		self.train_acc_path = []
		self.hold_acc_path = []
		self.test_acc_path = []

		self.train_loss_path = []
		self.hold_loss_path = []
		self.test_loss_path = []

		if self.adaptInitial:
			np.random.seed(0)
			self.biases = [np.random.normal(0,1.0//math.sqrt(i), size=(j,1)) for i,j in zip(sizes[:-1], sizes[1:])]
			self.weights = [np.random.normal(0,1.0/math.sqrt(j), size=(i,j)) for i,j in zip(sizes[1:], sizes[:-1])]
		else:
			np.random.seed(0)
			self.biases = [np.random.randn(j, 1) for j in sizes[1:]]
			self.weights = [np.random.randn(i, j) for i, j in zip(sizes[1:], sizes[:-1])]

	def Mini_GD(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None, T = 50, regularization = 'L2', reg_lam = 0.0001):
		partial = len(training_data) / 6 * 5
		hold_data = training_data[partial:]
		training_data = training_data[:partial]
		num_train = len(training_data)
		num_hold = len(hold_data)

		if test_data:
			num_test = len(test_data)
		self.regularization = regularization
		self.reg_lam = reg_lam

		for i in xrange(epochs):
			if self.shuffle_status:
				random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,num_train,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,learning_rate)

			learning_rate = learning_rate*1.0/(1+i*1.0/T)

			self.hold_acc_path.append(self.cal_Acc(hold_data, num_hold))
			self.hold_loss_path.append(self.cal_Loss(hold_data))
			self.train_acc_path.append(self.cal_Acc(training_data, num_train))
			self.train_loss_path.append(self.cal_Loss(training_data))
			if i > 2 and self.hold_acc_path[-3] > self.hold_acc_path[-2] and self.hold_acc_path[-2] > self.hold_acc_path[-1]:
				break
			if test_data:
				self.test_acc_path.append(self.cal_Acc(test_data, num_test))
				self.test_loss_path.append(self.cal_Loss(test_data))
				print "Epoch {0}: {1} {2}".format(i, self.cal_Acc(training_data, num_train), self.cal_Loss(training_data))
			else:
				print "Epoch {0} complete".format(i)

	def update_mini_batch(self, mini_batch, learning_rate):
		temp_b = [np.zeros(b.shape) for b in self.biases]
		temp_w = [np.zeros(w.shape) for w in self.weights]
		for x,y in mini_batch:
			y = np.reshape(y,(10,1))
			delta_b,delta_w = self.backprop(x,y)
			temp_b = [nb+dnb for nb,dnb in zip(temp_b,delta_b)]
			temp_w = [nw+dnw for nw,dnw in zip(temp_w,delta_w)]

		if self.gradientCheck:
			self.gradient_Check(x, y, delta_w, delta_b)
		if self.momentumUpdate:
			self.v_momentum_b = [self.mu*vb+learning_rate*tb/len(mini_batch)-self.reg_term(b) for vb,tb,b in zip(self.v_momentum_b, temp_b, self.biases)]
			self.v_momentum_w = [self.mu*vw+learning_rate*tw/len(mini_batch)-self.reg_term(w) for vw,tw,w in zip(self.v_momentum_w, temp_w, self.weights)]

			self.weights = [w+vw for w, vw in zip(self.weights, self.v_momentum_w)]
			self.biases = [b+vb for b, vb in zip(self.biases, self.v_momentum_b)]
		else:
			self.weights = [w+learning_rate*nw/len(mini_batch)-self.reg_term(w) for w, nw in zip(self.weights, temp_w)]
			self.biases = [b+learning_rate*nb/len(mini_batch)-self.reg_term(b) for b, nb in zip(self.biases, temp_b)]

	def backprop(self, x, y):
		temp_b = [np.zeros(b.shape) for b in self.biases]
		temp_w = [np.zeros(w.shape) for w in self.weights]
		# forword
		activation = x
		activations = [x]
		Zs = []
		for b,w in zip(self.biases[:-1], self.weights[:-1]):
			z = np.dot(w, activation) + b
			Zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)
		z = np.dot(self.weights[-1], activation) + self.biases[-1]
		Zs.append(z)
		activation = self.softmax(z)
		activations.append(activation)
		#backward
		delta = y-activations[-1]
		temp_b[-1] = delta
		temp_w[-1] = np.dot(delta, activations[-2].T)

		for l in xrange(2,self.numLayers):
			z = Zs[-l]
			derivative = self.sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].T, delta) * derivative
			temp_b[-l] = delta
			temp_w[-l] = np.dot(delta, activations[-l-1].T)
		return (temp_b, temp_w)

	def predict(self, x):
		for b,w in zip(self.biases[:-1], self.weights[:-1]):
			x = self.sigmoid(np.dot(w,x)+b)
		x = np.dot(self.weights[-1],x)+self.biases[-1]
		result = self.softmax(x)
		return result

	def cal_Acc(self, data, num_data):
		results = [(np.argmax(self.predict(x)), np.argmax(y)) for (x,y) in data]
		acc = sum(int(x==y) for (x,y) in results)
		acc = (acc + 0.0)/num_data
		return acc

	def cal_Loss(self,data):
		results = [np.log(self.predict(x)) * y.reshape(y.shape[0],1) for (x, y) in data] 
		return -np.sum(results) / len(results)

	def sigmoid(self, x):
		if self.activation_func == 'tanh':
			return 1.7159*np.tanh(2./3.*x)
		if self.activation_func == 'relu':
			return np.maximum(x, 0)
		return np.exp(x)/(1.0+np.exp(x))

	def sigmoid_prime(self, x):
		if self.activation_func == 'tanh':
			return 1.14393*(1-np.tanh(2./3.*x)**2)
		if self.activation_func == 'relu':
			return (x>0)*1
		return self.sigmoid(x)*(1-self.sigmoid(x))

	def softmax(self, x):
		return (np.exp(x)+0.0)/np.exp(x).sum(axis=0)[:,None]

	def reg_term(self, w):
		if self.regularization == 'L1':
			reg = np.ones(w.shape)
			reg[w<0] = -1
			return self.reg_lam*reg*w*1.0/(w.shape[0]*w.shape[1])
		elif self.regularization == 'L2':
			return 2*self.reg_lam*w*1.0/(w.shape[0]*w.shape[1])
		return np.zeros(w.shape)

	def predictForCheck(self, x, weight, bias):
		for b, w in zip(bias[:-1], weight[:-1]):
			x = self.sigmoid(np.dot(w, x) + b)
		x = np.dot(weight[-1], x) + bias[-1]
		result = self.softmax(x)
		return result

	def cal_E(self, x, t, weight, bias):
		result = np.dot(np.log(self.predictForCheck(x, weight, bias)).T, t)
		return float(result)

	def gradient_Check(self, x, t, delta_w, delta_b):
		error = 0.01
		weights = np.copy(self.weights)
		biases = np.copy(self.biases)
		w1 = np.random.randint(len(self.weights))
		w2 = np.random.randint(np.shape(self.weights[w1])[0])
		w3 = np.random.randint(np.shape(self.weights[w1])[1])
		changeWhich_w = [w1, w2, w3]

		b1 = np.random.randint(len(self.biases))
		b2 = np.random.randint(np.shape(self.biases[b1])[0])
		b3 = np.random.randint(np.shape(self.biases[b1])[1])
		changeWhich_b = [b1, b2, b3]

		weights[changeWhich_w[0]][changeWhich_w[1]][changeWhich_w[2]] += error
		E1 = self.cal_E(x, t, weights, biases)
		weights[changeWhich_w[0]][changeWhich_w[1]][changeWhich_w[2]] -= 2 * error
		E2 = self.cal_E(x, t, weights, biases)
		E_weight = (E1 - E2) * 0.5 / error

		biases[changeWhich_b[0]][changeWhich_b[1]][changeWhich_b[2]] += error
		E3 = self.cal_E(x, t, self.weights, biases)
		weights[changeWhich_b[0]][changeWhich_b[1]][changeWhich_b[2]] -= 2 * error
		E4 = self.cal_E(x, t, self.weights, biases)
		E_bias = (E3 - E4) * 0.5 / error

		weights_derivative_diff = np.abs(E_weight - delta_w[w1][w2][w3])
		biases_derivative_diff = np.abs(E_bias - delta_b[b1][b2][b3])
		print 'For Weights' + str(changeWhich_w) + ':'
		print 'difference between weights gradient and numerical approximation is', weights_derivative_diff
		print 'For Biases' + str(changeWhich_b) + ':'
		print 'difference between biases gradient and numerical approximation is', biases_derivative_diff

def plot3path(line1,label1,line2,label2,line3,label3,ylim,xlabel,ylabel,title):
    if ylim:
        plt.ylim(ylim)
	plt.plot(range(len(line1)),line1,label=label1)
    plt.plot(range(len(line2)),line2,label=label2)
    plt.plot(range(len(line3)),line3,label=label3)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()