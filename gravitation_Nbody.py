import sys
import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm

def integrate(ic, ti, p):
	rvto_list = ic
	m, Gc = p

	r=[]
	v=[]
	th=[]
	om=[]
	for i in range(m.size):
		r.append(rvto_list[4*i])
		v.append(rvto_list[4*i+1])
		th.append(rvto_list[4*i+2])
		om.append(rvto_list[4*i+3])

	sub={}
	for i in range(m.size):
		sub[M[i]] = m[i]
		sub[R[i]] = r[i]
		sub[R_dot[i]] = v[i]
		sub[THETA[i]] = th[i]
		sub[THETA_dot[i]] = om[i]
	sub['G'] = Gc

	diff_eqs = []
	for i in range(m.size):
		diff_eqs.append(v[i])
		diff_eqs.append(A[i].subs(sub))
		diff_eqs.append(om[i])
		diff_eqs.append(ALPHA[i].subs(sub))

	print(ti)

	return diff_eqs

#-----------------------------------------

N = 3

G, t = sp.symbols('G t')
M = sp.symbols('M0:%d'%N)
R = dynamicsymbols('R0:%d'%N)
THETA = dynamicsymbols('THETA0:%d'%N)

X = []
Y = []
vs = []
for i in range(N):
	X.append(R[i] * sp.cos(THETA[i]))
	Y.append(R[i] * sp.sin(THETA[i]))
	vs.append(X[i].diff(t,1)**2 + Y[i].diff(t,1)**2)

T = 0
for i in range(N):
	T += M[i] * vs[i] 
T *= sp.Rational(1,2)
T = sp.simplify(T)

dR = []
for i in range(N):
	temp = []
	for j in range(N):
		temp.append(sp.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2))
	dR.append(temp)

V = 0
for i in range(N):
	for j in range(N):
		if i != j: 
			V -= M[i] * M[j] / dR[i][j]
V *= G
V = sp.simplify(V)

L = T - V

R_dot = []
THETA_dot = []
A = []
ALPHA = []
for i in range(N):
	R_dot.append(R[i].diff(t, 1))
	THETA_dot.append(THETA[i].diff(t, 1))
	dLdR = L.diff(R[i], 1)
	dLdR_dot = L.diff(R_dot[i], 1)
	ddtdLdR_dot = dLdR_dot.diff(t, 1)
	dLA = ddtdLdR_dot - dLdR
	sol = sp.solve(dLA, R[i].diff(t, 2))
	A.append(sol[0])
	dLdTHETA = L.diff(THETA[i], 1)
	dLdTHETA_dot = L.diff(THETA_dot[i], 1)
	ddtdLdTHETA_dot = dLdTHETA_dot.diff(t, 1)
	dLALPHA = ddtdLdTHETA_dot - dLdTHETA
	sol = sp.solve(dLALPHA, THETA[i].diff(t, 2))
	ALPHA.append(sol[0])

#---------------------------------------------------

Gc = 1
mass_a, mass_b = [3, 5]
r_a, r_b = [4, 5]
v_a, v_b = [0, 0]
theta_a, theta_b = [0, 360]
omega_a, omega_b = [15, 45]
tf = 30
nfps = 30

initialize = "random"

cnvrt = np.pi/180
if initialize == "increment":
	m = np.linspace(mass_a, mass_b, N)
	r = np.linspace(r_a, r_b, N)
	v = np.linspace(v_a, v_b, N)
	theta = np.linspace(theta_a, theta_b, N) * cnvrt
	omega = np.linspace(omega_a, omega_b, N) * cnvrt
elif initialize == "random":
	rng=np.random.default_rng(92314311)
	m = (mass_b - mass_a) * np.random.rand(N) + mass_a
	r = (r_b - r_a) * np.random.rand(N) + r_a
	v = (v_b - v_a) * np.random.rand(N) + v_a
	theta = ((theta_b - theta_a) * np.random.rand(N) + theta_a) * cnvrt
	omega = ((omega_b - omega_a) * np.random.rand(N) + omega_a) * cnvrt
else:
	sys.exit("Initialization Routine Not Found. Choices are increment or random. Pick One.")

p = [m, Gc]
ic = []
for i in range(N):
	ic.append(r[i])
	ic.append(v[i])
	ic.append(theta[i])
	ic.append(omega[i])

nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rvto = odeint(integrate, ic, ta, args = (p,))

x = np.zeros((N, nframes))
y = np.zeros((N, nframes))
ke = np.zeros(nframes)
pe = np.zeros(nframes)
for i in range(nframes):
	ke_sub={}
	pe_sub={}
	for j in range(N):
		x[j][i] = X[j].subs({R[j]:rvto[i, 4 * j], THETA[j]:rvto[i, 4 * j + 2]})
		y[j][i] = Y[j].subs({R[j]:rvto[i, 4 * j], THETA[j]:rvto[i, 4 * j + 2]})
		ke_sub[M[j]] = m[j]
		ke_sub[R[j]] = rvto[i, 4 * j]
		ke_sub[R_dot[j]] = rvto[i, 4 * j + 1]
		ke_sub[THETA_dot[j]] = rvto[i, 4 * j + 3]
		pe_sub['G'] = Gc
		pe_sub[M[j]] = m[j]
		pe_sub[R[j]] = rvto[i, 4 * j]
		pe_sub[THETA[j]] = rvto[i, 4 * j + 2]
	ke[i] = T.subs(ke_sub)
	pe[i] = V.subs(pe_sub)

E = ke + pe

fig, a=plt.subplots()

#--------------------------------------------

sfac=1/50
xmax = [max(i) for i in x[:,]]
xmin = [min(i) for i in x[:,]]
ymax = [max(i) for i in y[:,]]
ymin = [min(i) for i in y[:,]]
xmax, xmin, ymax, ymin = [max(xmax), min(xmin), max(ymax), min(ymin)]
rfac = np.hypot(xmax - xmin, ymax - ymin) * sfac
rad = [i * rfac / max(m) for i in m]
xmax, ymax = np.array([xmax, ymax]) + 2 * max(rad)
xmin, ymin = np.array([xmin, ymin]) - 2 * max(rad)
clist = cm.get_cmap('gist_rainbow', N)

#--------------------------------------------
def run(frame):
	plt.clf()
	plt.subplot(211)
	for i in range(N):
		circle=plt.Circle((x[i][frame],y[i][frame]),radius=rad[i],fc=clist(i))
		plt.gca().add_patch(circle)
	plt.title("N-Body Orbital Dynamics (N=%i)" %N)
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('gravitation_Nbody.mp4', writer=writervideo)
plt.show()











