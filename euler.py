import control
import numpy as np

from matplotlib import pyplot as plt

step = lambda x: 1 if x >= 0 else 0

def discretization(tf, entry_signal=step, step=0.001, start=0, end=5, time=None):
	h = step if time is None else time[1] - time[0]
	time = np.arange(start, end, step) if time is None else time

	a_main = -np.array(tf.den[0][0][1:]) / tf.den[0][0][0]
	A = np.concatenate(([a_main], np.identity(len(a_main))[:-1]), axis=0)

	B = np.zeros(len(A))
	B[0] = 1

	c_zeros = len(a_main) - len(tf.num[0][0])
	C = np.array([[ *([0]*c_zeros), *tf.num[0][0]]]) / tf.den[0][0][0]

	x = [0] * len(A)
	y = []
	for n, t in enumerate(time):
		prox = x + h*( A @ x + B*entry_signal(t) )
		proy = C @ x

		x = prox
		y.append(proy[0])

	return time, y

def discretization_simple(kp, ki, kd, entry_signal=step, step=0.001, start=0, end=5, time=None):
	h = step if time is None else time[1] - time[0]
	time = np.arange(start, end, step) if time is None else time

	vs, vr, vc, ve, vt, vf = [0], [0], [0], [0], [0], [0]
	for k, t in enumerate(time):
		ve.append( entry_signal(t) - vs[k] )

		vc.append( kp*ve[k] + ki*h*sum(ve) + kd * (ve[k] - ve[k-1])/h )

		vr.append( .1/(.1 + h) *vr[k] + 10*h/(.1+h)*vc[-1] )

		vf.append( .4/(.4+h) *vf[k] + h/(.4+h)*vr[-1] )

		vt.append( 1/(1+h) *vt[k] + h/(1+h)*vf[-1] )

		vs.append( .01/(.01+h) *vs[k] + h/(.01+h)*vt[-1] )
	return time, vt[1:]

if __name__ == '__main__':
	s = control.TransferFunction.s
	tf = (2*s**2 + 8*s + 6) / (s ** 3 + 8 * s ** 2 + 16*s + 6)
	tf = (0.654 + .5389/s + .2458*s) * 10/(1 + .1*s) * 1/(1+.4*s) * 1/(1+s)

	tf = tf.feedback(1/(1 + .01*s))

	plot_time = np.arange(0, .6, .01)

	true_step = control.step_response(tf, plot_time)
	my_step = discretization(tf, time=plot_time)

	print('true step:',true_step[1][-5:])

	plt.plot(*true_step)
	plt.plot(*my_step, linewidth=2)
	plt.grid(True, linestyle='--')
	plt.legend(["Lib", "Mine"])
	plt.show()