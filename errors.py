def ISE(T, Y):
	error = 1 - Y
	return sum(error ** 2)

def IAE(T, Y):
	return sum(abs(1 - Y))

def ITSE(T, Y):
	error = 1 - Y
	return sum(T * error ** 2)

def ITAE(T, Y):
	error = 1 - Y
	return sum(T * abs(error))