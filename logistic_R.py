import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_reg(X, y, theta,b):
	x=X.T 
	plt.scatter(x[0,],x[1,])
	x3 = np.arange(-0.1,1.5,0.1)
	x2 = [-( b + theta[0][0]*x1)/theta[0][1] for x1 in x3]
	plt.plot(x3,x2,'r')
	plt.legend()
	plt.ylim([-0.1,1.1])
	plt.show()

	# 1.70474504 , 15.04062212 , -20.47216021

def sigmoid_fun(theta,X,b):
	return 1.0/(1+np.exp(-(np.dot(X,theta.T))))


def pred_values(theta, X,b): 
    ''' 
    function to predict labels 
    '''
    pred_prob = sigmoid_fun(theta, X,b) 
    pred_value = np.where(pred_prob >= .5, 1, 0) 
    return np.squeeze(pred_value)

# def loadCSV():
# 	fh=open('data_set.txt')
# 	lst1=list()
# 	lst2=list()
# 	lst3=list()
# 	for line in fh:
# 		line =line .strip('\n')
# 		nums=line.split('\t')
# 		lst1.append(float(nums[0]))
# 		lst2.append(float(nums[1]))
# 		lst3.append(float(nums[2]))
# 	x1=np.array(lst1).reshape(len(lst1),1)
# 	x2=np.array(lst2).reshape(len(lst1),1)
# 	x3=np.array(lst3).reshape(len(lst1),1)
# 	A=np.append(x1,x2,axis=1)
# 	# A=np.append(A,np.ones([x1.shape[0],1],dtype=float),axis=1)
# 	y=x3.reshape(100,1)
# 	# theta = np.zeros([3,1],dtype=float)
# 	return A,y

def loadCSV():
	data_set = pd.read_csv('data_set.csv')
	# data_set.as_matrix()
	df = np.array(data_set)
	print(df)
	X=df[:,:-1]
	y=df[:,-1:]
	return X,y



def normalize(A): 
	''' 
	function to normalize feature matrix, X 
	'''
	# print(A)
	mins = np.amin(A, axis = 0) 
	maxs = np.amax(A, axis = 0) 
	# print (maxs)
	# print(mins)
	rng = maxs - mins
	# print(rng) 
	# print(maxs-A)
	norm_X = 1 - ((maxs - A)/rng) 
	return norm_X 


if __name__ == "__main__":
	main()


def main():
	return 0;
	A,y  = loadCSV()
	print(A)

	X = normalize(A)

	# stacking columns wth all ones in feature matrix
	# X=np.append(X,np.ones([X.shape[0],1],dtype=float),axis=1)
	# print(X)

	# initial theta values all zeros !
	theta = np.zeros([1,X.shape[1]],dtype=float)
	b=0
	# alpha = 0.08  for data_set.txt
	alpha=0.01
	for itr in range(1000):
		a = sigmoid_fun(theta,X,b)
		dZ = a - y
		dW = np.dot(X.T,dZ)
		dtheta_zero = (1/X.shape[0])*(np.sum(dZ))
		theta = theta - alpha*(dW.T)
		b = b - alpha*dtheta_zero
	print(alpha)
	print(theta)
	print(b)
	print(X.shape[0])
	y_pred = pred_values(theta, X,b)
	print(y_pred)
	print(y.T)
	num = np.sum((y.T) == y_pred)
	print("Correctly predicted labels:", (num/y.shape[0])*100,"%")
	# plot_reg(X, y, theta,b)
	







	 
