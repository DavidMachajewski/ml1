import numpy as np
import math
class bias_variance_linear_regression:
	#comparison to Bishop notation: mean_value=h(x), functions= y^(l)(x)
	def __init__(self, functions, mean_value):
		self.functions=functions
		self.L=len(functions)
		self.mean_value=mean_value
		
	#mean prediction
	def mean_prediction(self,x):
		return (1/self.L)*np.sum([f.predict(np.array([x])) for f in self.functions])
		
	def bias_squared_regression(self,data):
		return (1/len(data))*np.sum([math.pow(self.mean_prediction(x)-self.mean_value(x),2) for x in data])
		
		
		
		