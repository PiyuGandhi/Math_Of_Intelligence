import numpy as np # Math operations
import matplotlib.pyplot as plt # Plotting Operations

# Step -1 : Defining the data

####### Training Data #########
# Input array (Features) - Every row has [x-coordinate , y-coordinate , Bias]

X = np.array([
	[-2,4,-1],
	[4,1,-1],
	[1,6,-1],
	[2,4,-1],
	[6,2,-1],
])

# Cooresponding Labels for the features

y = np.array([-1,-1,1,1,1])

# Step -2 : Plotting our data

for d, sample in enumerate(X):

	if d < 2: # Negative Samples
		plt.scatter(sample[0],sample[1],s=120,marker="_",linewidths=2)

	else: # Positive Samples
		plt.scatter(sample[0],sample[1],s=120,marker="+",linewidths=2)



######### Testing Data #########

plt.scatter(2,2,s=120,marker="_",linewidths=2,color ='black')
plt.scatter(4,3,s=120,marker="+",linewidths=2,color='black')

plt.title("Sample hyperplane and points")
# Sample Hyperplane (Just an example guess of a line)
plt.plot([-2,6],[6,0.5])
# Uncomment the line below to see the visualized result(Graph)
plt.show()


################ MATH BEHIND IT ###########################
'''
Machine Learning is about optimizing.
For optimizing any function(or anything) , 
we need :- 
			1. A Goal function (To be optimized)
			2. A Loss/Error function (Which tells us how we are better than the previous state)

Here:- 

1. GOAL FUNCTION:-  
	G(let's say) = min( regulization_function + loss_function)
	
	Here,
		regulization_function = Tells us how good a plane fits ( If regulization_function is HIGH , OVERFIT 
																------------------------ LOW , UNDERFIT)
		loss_function = tells us the error/gap between the plane and the points


2. REGULIZATION FUNCTION:-
	R = lambda * || w * w ||

	Here,
		lambda = learning rate
		w = weight
	
3. LOSS FUNTION:- 
	We are going to be using "Hinge-Loss" function -> maximizes the gap between two classes of data

	L(x,y,f(x)) = (1 - y*f(x))+ ,

		Here,
			y = Original Value
			f(x) = Predicted value (f(x) = current_weight*x + current_bias)
			f(x) = y(i)(x(i),w) (w = weight)

				    ______ 0 , if y*f(x) >= 1
	L(x,y,f(x))  = /
				   \______ 1-y*f(x) , else

	

		---------HOW TO OPTIMIZE---------

3. OPTIMIZATION STRATEGY:-
	The easiest way to optimize a plane is to go towards it's derivative (of error).
	We will take the derivatives of both regulization_function and loss_function wrt. w (weight)
	
	derivative(regulization_function) = 2 * lambda * w
								 _______ 0 , if y(i)(x(i),w) >= 1
	derivative(loss_function) = /
								\_______ -y(i) * x(i) , else
	


 	------------LOGIC/UPDATE RULE FOR WEIGHTS-------------

 Now, there are "2" possible cases on a sample :-

 1. MISCLASSIFICATION:- ( When the hyperplane doesn't pass through that point)

 	In that case,
 		Update rule for weights:-
		
		w (new) = w + (lambda)*( (y(i) * x(i)) - 2 * (n) * w)
		
		Here,
			w = weight
			lambda = learning rate = 1/(no of iterations) => (As the iteration increases, learning rate decreases)

2. CLASSIFICATION:- ( When the hyperplane passes through the point)
	
	In that case,
		Update rule for weights:-

		w (new) = w + (lambda) * ( -2 * (n) * w) 
'''

# Stochastic Gradient Descent 

def svm_sgd_plot(X,Y):

	# Initialize with zeroes (or random , your choice)
	w = np.zeros(len(X[0]))
	# The learning rate
	lamb = 1
	# Iterations to run for
	iterations = 100000

	# Store the misclassifications so to visualize how they change over time
	misclassifications = []

	# Training Part , Gradient Descent

	for epoch in range(1,iterations):
		miss = 0
		for i,x in enumerate(X):
			if Y[i]*np.dot(X[i],w) < 1: # Misclassification
				w = w + (lamb) * ( (Y[i]*X[i]) + (-2 * (1/epoch) * w))
				miss = 1
			else: # Classified correctly
				w = w + (lamb) * (-2 * (1/epoch) * w)
		misclassifications.append(miss)

	# Plotting rate of classification errors during the training
	plt.plot(misclassifications , '|')
	plt.ylim(0.5,1.5)
	plt.axes().set_yticklabels([])
	plt.xlabel('Iteration')
	plt.ylabel('Misclassified')
	plt.title("Iterations v/s Misclassified Data")
	# Uncomment to show the plot
	plt.show()

	return w


if __name__ == "__main__":
	w = svm_sgd_plot(X,y) # Training and finding final weights


	# Plotting the trainig and testing data again for final plane visualization
	for d, sample in enumerate(X):

		if d < 2: # Negative Samples
			plt.scatter(sample[0],sample[1],s=120,marker="_",linewidths=2)

		else: # Positive Samples
			plt.scatter(sample[0],sample[1],s=120,marker="+",linewidths=2)

	plt.scatter(2,2,s=120,marker="_",linewidths=2,color ='black')
	plt.scatter(4,3,s=120,marker="+",linewidths=2,color='black')

	

	# Plotting the hyperplane
	x2 = [w[0], w[1], -w[1], w[0]]
	x3 = [w[0], w[1], w[1], -w[0]]

	x2x3 = np.array([x2,x3])
	X,Y,U,V = zip(*x2x3)
	
	

	ax = plt.gca()
	ax.quiver(X,Y,U,V, scale= 1, color= 'blue')
	plt.title("Final line after training (Black points are Testing)")
	plt.show()