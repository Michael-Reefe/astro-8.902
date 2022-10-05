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
def getAcc( pos, mass, G, softening ):
	"""
    Calculate the acceleration on each particle due to Newton's Law 
	pos  is an N x 3 matrix of positions
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	softening is the softening length
	a is N x 3 matrix of accelerations
	"""
	# positions r = [x,y,z] for all particles
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r^3 for all particle pairwise particle separations 
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
	for gi in nb.prange(inv_r3.shape[0]):
		for gj in nb.prange(inv_r3.shape[1]):
			if inv_r3[gi, gj] > 0:
				inv_r3[gi, gj] = inv_r3[gi, gj]**(-1.5)

	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))

	return a

@nb.njit
def getEnergy( pos, vel, mass, G ):
	"""
	Get kinetic energy (KE) and potential energy (PE) of simulation
	pos is N x 3 matrix of positions
	vel is N x 3 matrix of velocities
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	KE is the kinetic energy of the system
	PE is the potential energy of the system
	"""
	# Kinetic Energy:
	KE = 0.5 * np.sum( mass * vel**2 )


	# Potential Energy:

	# positions r = [x,y,z] for all particles
	x = pos[:,0:1]
	y = pos[:,1:2]
	z = pos[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r for all particle pairwise particle separations 
	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	for gi in nb.prange(inv_r.shape[0]):
		for gj in nb.prange(inv_r.shape[1]):
			if inv_r[gi, gj] > 0:
				inv_r[gi, gj] = 1.0/inv_r[gi, gj]

	# sum over upper triangle, to count each interaction only once
	PE = G * np.sum(np.triu(-(mass*mass.T)*inv_r,1))

	# calculate the normalized virital
	virnorm = (2*KE + PE) / (KE + PE)
	
	return virnorm

@nb.njit
def estimate_freefall( N, pos, vel, mass, G ):
	"""
	Estimate the free-fall time
	"""
	# positions r = [x,y,z]
	x = pos[:, 0:1]
	y = pos[:, 1:2]
	z = pos[:, 2:3]
	# velocities v = [vx,vy,vz]
	vx = vel[:, 0:1]
	vy = vel[:, 1:2]
	vz = vel[:, 2:3]
	# matrix stores all pairwise separations r_ij = r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z
	# matrix stores all pairwise velocities v_ij = v_j - v_i
	dvx = vx.T - vx
	dvy = vy.T - vy
	dvz = vz.T - vz
	# store r_ij for all particle pairs
	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	for gi in nb.prange(inv_r.shape[0]):
		for gj in nb.prange(inv_r.shape[1]):
			if inv_r[gi, gj] > 0:
				inv_r[gi, gj] = 1.0/inv_r[gi, gj]
	
	# store v_ij for all particle pairs
	v = np.sqrt(dvx**2 + dvy**2 + dvz**2)

	# store phi^2 per unit mass per unit G for each particle
	phi2 = np.sum(mass.T * inv_r, axis=0)**2
	# store phidot^2 per unit mass per unit G for each particle
	phidot2 = np.sum(mass.T * inv_r**2 * v, axis=0)**2

	# calculate t_free-fall
	t_ff = np.sqrt(1/N * np.sum(phi2 / phidot2))

	return t_ff


def main():
	""" N-body simulation """
	
	# Simulation parameters
	N         = 200    # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 50.0   # time at which simulation ends
	dt        = 0.01   # timestep
	softening = 0.1    # softening length
	G         = 1.0    # Newton's Gravitational Constant
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	np.random.seed(17)            # set the random number generator seed
	
	mass = 40.0*np.ones((N,1))/N  # total mass of particles is 40
	pos  = np.random.randn(N,3)   # randomly selected positions and velocities
	vel  = np.random.randn(N,3)

	# Offset half of the particles's x positions
	Nhalf = N // 2
	pos[:Nhalf, 0] -= 10
	pos[Nhalf:, 0] += 10
	
	# Convert to Center-of-Mass frame
	vel -= np.mean(mass * vel,0) / np.mean(mass)
	
	# calculate initial gravitational accelerations
	acc = getAcc( pos, mass, G, softening )
	
	# calculate initial energy of system
	virnorm  = getEnergy( pos, vel, mass, G )

	# rough estimate of free-fall time
	t_ff = estimate_freefall( N, pos, vel, mass, G )

	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# save energies, particle orbits for plotting trails
	pos_save = np.zeros((N,3,Nt+1))
	pos_save[:,:,0] = pos
	vir_save = np.zeros(Nt+1)
	vir_save[0] = virnorm
	t_all = np.arange(Nt+1)*dt

	# make ACTUAL matplotlib animations
	# prep figure
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])

	xx, yy = pos_save[:,0,0], pos_save[:,1,0]
	sca = ax1.scatter([], [], color=[.7,.7,1], s=1)
	sca0 = ax1.scatter(xx, yy, color='blue', s=5)
	eng = ax2.scatter([], [], label='$t_{ff} \sim %.2f$' % t_ff, color='red', s=1)

	ax1.set(xlim=(-12, 12), ylim=(-12, 12))
	ax1.set_aspect('equal', 'box')
	ax1.set_xticks(np.arange(-12, 16, 4))
	ax1.set_yticks(np.arange(-12, 16, 4))

	ax2.set(xlim=(0, tEnd), ylim=(-1, 1))
	ax2.set_aspect(tEnd/8, 'box')

	ax2.set_xlabel('time')
	ax2.set_ylabel('virial')
	# ax2.legend(loc='upper right')

	FFMpegWriter = ani.writers['ffmpeg']
	writer = FFMpegWriter(fps=30)

	with writer.saving(fig, 'nbody.mp4', 100):
		
		# Simulation Main Loop
		for i in tqdm.trange(Nt):
			# (1/2) kick
			vel += acc * dt/2.0
			
			# drift
			pos += vel * dt
			
			# update accelerations
			acc = getAcc( pos, mass, G, softening )
			
			# (1/2) kick
			vel += acc * dt/2.0
			
			# update time
			t += dt
			
			# get energy of system
			virnorm = getEnergy( pos, vel, mass, G )
			
			# save energies, positions for plotting trail
			pos_save[:,:,i+1] = pos
			vir_save[i+1] = virnorm
			
			# only save every 5 frames (for efficiency)
			if i % 5 == 0:
				# plot in real time every 1/10th of the simulation
				xx = pos_save[:,0,max(i-50,0):i+1]
				yy = pos_save[:,1,max(i-50,0):i+1]
				sca.set_offsets(np.c_[xx.ravel(), yy.ravel()])
				sca0.set_offsets(np.c_[pos[:,0], pos[:,1]])
				eng.set_offsets(np.c_[t_all[vir_save != 0],vir_save[vir_save != 0]])
				
				writer.grab_frame()

			# save every 1/10th of simulation
			if i % (Nt//10) == 0 or i == (Nt-1):
				plt.savefig(f"nbody_{i}.png", dpi=300, bbox_inches='tight')
	
	plt.close()

	return 0
	


  
if __name__== "__main__":
  main()
