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
def estimate_freefall( N, pos, vel, mass ):
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
	N         = 242    # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 1000.0  # time at which simulation ends
	dt        = 0.01   # timestep
	softening = 0.0    # softening length TURN THIS OFF
	T         = 10                  # Gravity oscillation period
	G         = lambda t: 1.0 + np.sin(2*np.pi*t/T)
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	pos = np.zeros((N,3))
	vel = np.zeros((N,3))
	mass = np.ones((N,1)) / N / 100

	r_scale = 12        # radial separation of each ring

	# Central, massive, at rest particles
	mass[0] = 1
	pos[0,:] = (0,0,0)
	vel[0,:] = (0,0,0)

	# mass[N//2] = 1
	# pos[N//2,:] = (5*r_scale,0,0)
	# r_m = 1/np.sqrt(2*r_scale)+0.05
	# vel[N//2,:] = (-r_m * 0.8,r_m * 0.5,0)

	inc = (0, np.pi/4)       # inclination of the disk orientation, 0 = face-on in the x-y plane, pi/2 = edge-on

	n_ring = 6 * np.arange(2, 7, 1)
	r_percent = np.linspace(0.2, 0.6, len(n_ring))

	for n_gal in range(1):
		for n in range(len(n_ring)):

			# Radial distance of the particle ring
			r_ring = r_percent[n] * r_scale
			# Angular separation between each particle
			phi = 2*np.pi / n_ring[n]
			# lower index on matrices based on previous n_rings, offset by 1 because of central mass
			low = 1 + int(np.sum(n_ring[0:n])) + n_gal * (np.sum(n_ring) + 1)
			# upper index '' '' ''
			upp = 1 + int(np.sum(n_ring[0:n+1])) + n_gal * (np.sum(n_ring) + 1)

			# convert spherical to x,y,z coordinates, in the x-y plane
			x = r_ring * np.cos(phi * np.arange(n_ring[n]))
			y = r_ring * np.sin(phi * np.arange(n_ring[n]))
			z = np.zeros(n_ring[n])

			# rotate y-z plane by some inclination
			pos[low:upp, 0] = pos[n_gal*N//2,0] + x
			pos[low:upp, 1] = pos[n_gal*N//2,1] + y * np.cos(inc[n_gal]) - z * np.sin(inc[n_gal])
			pos[low:upp, 2] = pos[n_gal*N//2,2] + y * np.sin(inc[n_gal]) + z * np.cos(inc[n_gal])

			# initial velocities should be set such that centripetal force = gravitational force from the central body
			v_ring = np.sqrt(G(t) * mass[0] / r_ring)
			vx = -v_ring * np.sin(phi * np.arange(n_ring[n]))
			vy = v_ring * np.cos(phi * np.arange(n_ring[n]))
			vz = np.zeros(n_ring[n])

			# apply the same inclination to the velocities
			vel[low:upp, 0] = vel[n_gal*N//2,0] + vx
			vel[low:upp, 1] = vel[n_gal*N//2,1] + vy * np.cos(inc[n_gal]) - vz * np.sin(inc[n_gal])
			vel[low:upp, 2] = vel[n_gal*N//2,2] + vy * np.sin(inc[n_gal]) + vz * np.cos(inc[n_gal])

	# Stay in the rest frame of galaxy 1
	# vel -= np.mean(mass * vel,0) / np.mean(mass)
	
	# calculate initial gravitational accelerations
	acc = getAcc( pos, mass, G(t), softening )
	
	# calculate initial energy of ONLY galaxy 1
	virnorm  = getEnergy( pos[0:N, :], vel[0:N, :], mass[0:N], G(t) )

	# rough estimate of free-fall time
	t_ff = estimate_freefall( N, pos, vel, mass )

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
	sca0 = ax1.scatter(xx, yy, color='blue', s=2)
	eng = ax2.scatter([], [], label='$t_{ff} \sim %.2f$' % t_ff, color='red', s=1)

	ax1.set(xlim=(-2*r_scale, 2*r_scale), ylim=(-2*r_scale, 2*r_scale))
	ax1.set_aspect('equal', 'box')
	# ax1.set_xticks(np.arange(-12, 16, 4))
	# ax1.set_yticks(np.arange(-12, 16, 4))

	ax2.set(xlim=(0, tEnd), ylim=(-3, 3))
	# ax2.set_aspect(tEnd/8, 'box')

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
			acc = getAcc( pos, mass, G(t), softening )
			
			# (1/2) kick
			vel += acc * dt/2.0
			
			# update time
			t += dt
			
			# get energy of only galaxy 1
			virnorm = getEnergy( pos[0:N, :], vel[0:N, :], mass[0:N], G(t) )
			
			# save energies, positions for plotting trail
			pos_save[:,:,i+1] = pos
			vir_save[i+1] = virnorm
			
			# only save every X frames (for efficiency)
			if i % 100 == 0:
				# plot in real time
				xx = pos_save[:,0,max(i-50,0):i+1] - pos_save[0,0,max(i-50,0):i+1]
				yy = pos_save[:,1,max(i-50,0):i+1] - pos_save[0,1,max(i-50,0):i+1]
				sca.set_offsets(np.c_[xx.ravel(), yy.ravel()])
				sca0.set_offsets(np.c_[pos[:,0]-pos[0,0], pos[:,1]-pos[0,1]])
				eng.set_offsets(np.c_[t_all[vir_save != 0],vir_save[vir_save != 0]])
				
				writer.grab_frame()

			# save every 1/10th of simulation
			if i % (Nt//10) == 0 or i == (Nt-1):
				plt.savefig(f"nbody_{i}.png", dpi=300, bbox_inches='tight')
	
	plt.close()

	return 0
	


  
if __name__== "__main__":
  main()
