import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numba as nb
import tqdm

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity

Modified by Michael Reefe (2022)
"""

@nb.njit
def getAcc( pos, mass, G, alpha, rho0, r0, l ):
	"""
    Calculate the acceleration on each particle due to Newton's Law 
	pos  is an N x 3 matrix of positions
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	softening is the softening length
	a is N x 3 matrix of accelerations
	"""
	u = pos[0]
	if alpha == 0:
		acc = 4/3 * np.pi * G * rho0 / (l**2 * u**3) - u
	elif alpha == 1:
		acc = 2 * np.pi * G * rho0 * r0 / (l**2 * u**2) - u
	elif alpha == 2:
		acc = 4 * np.pi * G * rho0 * r0**2 / (l**2 * u) - u	
	elif alpha == 3:
		acc = 4 * np.pi * G * rho0 * r0**3 / l**2 * (np.log(1/(u * r0)) - 1) - u
	else:
		raise ValueError
	
	return acc

@nb.njit
def getEnergy( pos, vel, mass, G, alpha, rho0, r0, l ):
	"""
	Get kinetic energy (KE) and potential energy (PE) of simulation
	pos is N x 3 matrix of positions
	vel is N x 3 matrix of velocities
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	KE is the kinetic energy of the system
	PE is the potential energy of the system
	"""
	r = 1/pos[0]
	dudt = l * pos[0]**2 * vel[0]
	drdt = -1/pos[0]**2 * dudt
	dpdt = vel[1]

	KE = 0.5 * mass * (drdt**2 + r**2 * dpdt**2)

	if alpha == 0:
		PE = 2/3 * np.pi * G * rho0 * r**2 * mass
	elif alpha == 1:
		PE = 2 * np.pi * G * rho0 * r0 * r * mass
	elif alpha == 2:
		PE = 4 * np.pi * G * rho0 * r0**2 * np.log(r/r0) * mass
	elif alpha == 3:
		PE = -4 * np.pi * G * rho0 * r0**3/r * np.log(r/r0) * mass
	else:
		raise ValueError
	
	return (2*KE+PE)/(KE+PE)

def main():
	""" N-body simulation """
	
	# Simulation parameters
	N         = 1           # Number of particles
	t         = 0           # current time of the simulation in TU
	tEnd      = 5000.0        # time at which simulation ends in TU (1 TU ~ 14.908 Myr)
	# tEnd      = 25.0        # time at which simulation ends in TU (1 TU ~ 14.908 Myr)
	dt        = 0.01        # time step
	alpha     = 3
	rho0      = 0.427       # Msun / pc^3, density scale (Binney & Tremaine)
	r0        = 1000.0      # pc, radius scale (Binney & Tremaine)
	G         = 1.0         # Newton's Gravitational Constant
	plotRealTime = True     # switch on for plotting as the simulation goes along
	
	np.random.seed(17)

	# Generate Initial Conditions	
	mass = 1                        # total mass of particle is 1 Msun
	# pos = np.array([1/1e3,  0])     # initial position in ([u] = 1/pc, [psi] = rad)
	pos = np.array([1/7.3e3, 0])
	# vel = np.array([-1/1e6, 1])     # initial velocity in ([du/dpsi] = 1/pc/rad, [dpsi/dt] = rad/TU)
	vel = np.array([-1/1e6, 0.1])

	l = 1/pos[0]**2 * vel[1]   # (constant) angular momentum
	
	# calculate initial gravitational accelerations
	acc = getAcc( pos, mass, G, alpha, rho0, r0, l )
	
	# calculate initial energy of system
	virnorm  = getEnergy( pos, vel, mass, G, alpha, rho0, r0, l )

	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# save energies, particle orbits for plotting trails
	pos_save = np.zeros((2,Nt+1))
	pos_save[:,0] = pos
	vir_save = np.zeros(Nt+1)
	vir_save[0] = virnorm
	t_all = np.arange(Nt+1)*dt

	# make ACTUAL matplotlib animations
	# prep figure
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.6)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])

	rr, pp = 1/pos_save[0,0], pos_save[1,0]
	xx, yy = rr * np.cos(pp), rr * np.sin(pp)
	ln, = ax1.plot([], [], color=[.7,.7,1], lw=0.5, alpha=1)
	sca0 = ax1.scatter(xx/1e3, yy/1e3, color='blue', s=5)
	eng = ax2.scatter([], [], color='red', s=1)

	ax1.set(xlim=(-10, 10), ylim=(-10, 10))
	ax1.set_aspect('equal', 'box')
	ax1.set_xticks(np.arange(-12, 15, 3))
	ax1.set_yticks(np.arange(-12, 15, 3))
	ax1.set_xlabel('$x$ (kpc)')
	ax1.set_ylabel('$y$ (kpc)')

	TU = 14.908
	ax2.set(xlim=(0, tEnd*TU), ylim=(-3, 3))
	# ax2.set_aspect(tEnd/8, 'box')

	ax2.set_xlabel('time (Myr)')
	ax2.set_ylabel('virial')
	# ax2.legend(loc='upper right')

	FFMpegWriter = ani.writers['ffmpeg']
	writer = FFMpegWriter(fps=30)

	with writer.saving(fig, 'nbody.mp4', 100):
		
		# Simulation Main Loop
		for i in tqdm.trange(Nt):

			# convert timestep to angle step
			dp = l * pos[0]**2 * dt

			# (1/2) kick to du/dpsi
			vel[0] += acc * dp/2.0
			
			# drift in u
			pos[0] += vel[0] * dp

			# conserve angular momentum
			vel[1] = l * pos[0]**2

			# drift in psi
			pos[1] += dp

			# update accelerations
			acc = getAcc( pos, mass, G, alpha, rho0, r0, l )
			
			# (1/2) kick to du/dpsi
			vel[0] += acc * dp/2.0
			
			# update time
			t += dt
			
			# get energy of system
			virnorm = getEnergy( pos, vel, mass, G, alpha, rho0, r0, l )
			
			# save energies, positions for plotting trail
			pos_save[:,i+1] = pos
			vir_save[i+1] = virnorm
			
			# only save every 5 frames (for efficiency)
			if i % 5 == 0:
				# plot in real time every 1/10th of the simulation
				rr = 1/pos_save[0,0:i+1]
				pp = pos_save[1,0:i+1]
				xx = rr * np.cos(pp)
				yy = rr * np.sin(pp)
				ln.set_data(xx.ravel()/1e3, yy.ravel()/1e3)
				sca0.set_offsets(np.c_[1/pos[0]*np.cos(pos[1])/1e3, 1/pos[0]*np.sin(pos[1])/1e3])
				eng.set_offsets(np.c_[t_all[vir_save != 0]*TU,vir_save[vir_save != 0]])
				
				writer.grab_frame()

			# save every 1/10th of simulation
			if i % (Nt//10) == 0 or i == (Nt-1):
				plt.savefig(f"nbody_{i}.png", dpi=300, bbox_inches='tight')
	
	plt.close()

	return 0
	
# constant steps in angle vs. time means the animated plot does not move linearly in time but in angle
# so the particle appears faster when closer in even though it should be moving slower

  
if __name__== "__main__":
  main()
