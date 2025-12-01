import numpy as np
import tensorflow as tf
import math

# Modified version of original method for testing M2 in run_model_all
# LOAD and TEST ---------------------------- #
def load_and_test_model(model_path, file_x1, file_x, file_y, file_ds):
	# modified to treat input as loaded files
	x1_test = file_x1
	x_test = file_x
	y_test = file_y
	ds_test = file_ds
	
	# Normalizing y
	mmin = 0.0
	mmax = 1.0
	c = 0
	y_test = nor_y_ab(y_test,c,mmin,mmax)

	freq = count_frequency_1(y_test,10,1)
	plot_freq(freq)
	
	# Loading model
	mymodel = tf.keras.models.load_model(model_path)
	
	# Testing
	print("Model prediction...")
	print("x_test: ",x_test.shape)
	print("x1_test: ",x1_test.shape)
	y_pred = mymodel.predict([x_test, x1_test])
	print("y_test: ",y_test.shape)
	print("y_pred: ",y_pred.shape)	
	
	# Error
	y_pred = y_pred.reshape(-1) 
	diff = abs(y_test - y_pred)
	print ("Norm MAE: ", np.mean(diff))
	rma0, mape0, wmape0, freq0, num0, mae_zero0, freq_zero0, num_zero0, wmapet0, out0, out_zero0 = mape_error_zero (y_test, y_pred)
	print ("Norm MAPE/WMAPE/Freq/Outliers/Tot_non_zero: ", mape0, wmape0, freq0, out0, num0)
	print ("Norm RMA non zero: ", rma0)
	print ("Norm MAE/WMAPETOT/Freq/Outliers/Tot_zero: ", mae_zero0, wmapet0, freq_zero0, out_zero0,  num_zero0)

	# ERROR evaluation DEnormalized VALUES --------------------------------
	#
	y_test_den = denorm_y_ab(y_test,c,mmin,mmax)
	y_pred_den = denorm_y_ab(y_pred,c,mmin,mmax)

	print ("Actual min/max: ", np.amin(y_test),"/", np.amax(y_test))
	print ("Prediction min/max: ", np.amin(y_pred),"/", np.amax(y_pred))
	diff_den = abs(y_test_den - y_pred_den)
	print ("MAE: ", np.mean(diff_den))
	rma, mape, wmape, freq, num, mae_zero, freq_zero, num_zero, wmape_tot, outliers, outliers_zero = mape_error_zero (y_test_den, y_pred_den)
	print ("MAPE/WMAPE/Freq/Outliers/Tot_non_zero: ", mape, wmape, freq, outliers, num)
	print ("RMA non zero: ", rma)
	print ("MAE/WMAPETOT/Freq/Outliers/Tot_zero: ", mae_zero, wmape_tot, freq_zero, outliers_zero, num_zero)
	return y_test_den, y_pred_den, rma, mape, wmape, mae_zero, freq, freq_zero, wmape_tot, outliers, outliers_zero

# Helper function
def mape_error_zero (y, predict):
	delta_zero = 0.0
	zero = 0
	non_zero = 0
	outliers = 0
	outliers_zero = 0
	delta = 0.0
	delta_w = 0.0
	den_w = 0.0
	p = 0
	rma = 0.0
	freq_zero = np.zeros((7), dtype=int)
	freq = np.zeros((7), dtype=int)
	for i in range(y.shape[0]):
		val = predict[i]
		if (val < 0.0):
			val = 0.0
		if (y[i] == 0.0):
			zero += 1
			delta_zero += val
			if (val == 0.0):
				freq_zero[6] += 1
			elif (val < 0.000001):
				freq_zero[5] += 1
			elif (val < 0.00001):
				freq_zero[4] += 1
			elif (val < 0.0001):
				freq_zero[3] += 1
			elif (val < 0.001):
				freq_zero[2] += 1
			elif (val < 0.01):
				freq_zero[1] += 1
			elif (val < 0.1):
				freq_zero[0] += 1
			else:
				outliers_zero += 1
			if (zero < 10 and val > 0.1):
				print(y[i], val)
		else:
			non_zero += 1
			delta += abs(y[i] - val)/y[i]
			if (non_zero < 10 and (abs(y[i] - val)/y[i]) > 100):
				print("High relative error: ",abs(y[i] - val)/y[i])
			delta_w += abs(y[i] - val)
			a = abs(predict[i]/y[i])
			if (a < 1.0):
				a = 1/a
			rma += a
			den_w += y[i]
			if (abs(y[i] - predict[i])/y[i] < 0.00001):
				freq[6] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.0001):
				freq[5] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.001):
				freq[4] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.01):
				freq[3] += 1
			elif (abs(y[i] - predict[i])/y[i] < 0.1):
				freq[2] += 1
			elif (abs(y[i] - predict[i])/y[i] < 1):
				freq[1] += 1
			elif (abs(y[i] - predict[i])/y[i] < 10):
				freq[0] += 1
			else:
				outliers += 1
	
			if (p < 10 and abs(y[i] - predict[i])/y[i] > 50):
				print(y[i], predict[i])
				p += 1
	if (zero==0):
		zero = -1
	return rma/non_zero, delta/non_zero, delta_w/den_w , freq, non_zero, delta_zero/zero, freq_zero, zero, (delta_w+delta_zero)/den_w, outliers, outliers_zero

# Helper functions extracted from other files
def plot_freq(f):
  max = 0
  for k in range(f.shape[0]):
    if (f[k] > max):
      max = f[k]
  for k in range(f.shape[0]):
    print('{0:5d} {1} {2}'.format(k, '+' * int(f[k]/max*50), f[k]))
    
def nor_y_ab(y,c,min,max):
	# c = 0: normalization MIN-MAX
	# c > 0: each value x is converted by applying the logarithic function new_x = log(1+c*x)
	# min = -1: the minimum is computed
	# max = -1: the maximum is computed
	print("Normalizing y with AB approach...")
	minimum = 0.0
	maximum = 1.0
	y = np.double(y)
	if (c > 0):
		y_norm = np.log(1 + c * y)
	else:
		y_norm = y
	if (min == -1):
		minimum = np.amin(y_norm, axis=(0))
	else:
		if (c > 0):
			minimum = math.log(1+c*min)
		else:
			minimum = min
	
	if (max == -1):
		maximum = np.amax(y_norm, axis=(0))
	else:
		if (c > 0):
			maximum = math.log(1+c*max)
		else:
			maximum = max
	y_norm = ( y_norm - minimum ) / ( maximum - minimum )
	# 
	print("MIN: if (c>0) them log(1+c*min) else min: ", minimum)
	print("MAX: if (c>0) them log(1+c*max) else max: ", maximum)
	return y_norm

def denorm_y_ab(y_nor, c, min, max):
	print("Denormalizing y..")
	y_nor = np.double(y_nor)
	min = np.double(min)
	max = np.double(max)
	if (c > 0):
		min_log = math.log(1+c*min)
		max_log = math.log(1+c*max)
		delta = max_log - min_log
		y = np.exp(y_nor * delta + min_log)
		y = (y - 1)/c
	else:
		delta = max - min
		y = y_nor * delta + min
	return y

def count_frequency_1(a,numClass,max):
	freq = np.zeros(numClass+1)
	for i in range(a.shape[0]):
		if (a[i] == 0):
			index = 0
		else:
			index = math.ceil(a[i]/max*numClass)
		if (index >= numClass+1):
			index = numClass
		freq[index] += 1
	return freq