import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numba as nb
from scipy.io import wavfile
from midiutil.MidiFile import MIDIFile
from midi2audio import FluidSynth
import tqdm
import subprocess

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity

Modified by Michael Reefe (2022)
Idea: turn orbit frequencies into musical notes -> plot the Fourier transform of the period
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
	

# @nb.njit
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

	N = len(x)
	f = np.zeros(N)
	for n_gal in range(2):
		# reduced mass
		m_red = 1/(1/mass[n_gal] + 1/mass[n_gal*N//2:(n_gal+1)*N//2])
		# relative velocity
		vx = vel[n_gal*N//2:(n_gal+1)*N//2,0:1]-vel[n_gal*N//2,0:1]
		vy = vel[n_gal*N//2:(n_gal+1)*N//2,1:2]-vel[n_gal*N//2,1:2]
		vz = vel[n_gal*N//2:(n_gal+1)*N//2,2:3]-vel[n_gal*N//2,2:3]
		v = np.sqrt(vx**2 + vy**2 + vz**2)
		# specific angular momentum
		L = (1/inv_r[n_gal*N//2,n_gal*N//2:(n_gal+1)*N//2]) * v.T
		# specific energy of individual orbits
		E = (0.5 * mass[n_gal*N//2:(n_gal+1)*N//2].T * v.T**2 - G * mass[n_gal*N//2:(n_gal+1)*N//2].T * mass[n_gal*N//2] * inv_r[n_gal*N//2,n_gal*N//2:(n_gal+1)*N//2]) / m_red.T
		# eccentricity
		sv = 1 + 2*E*L**2/(G * mass[n_gal*N//2])**2
		for j in nb.prange(sv.shape[1]):
			sv[0,j] = np.max(np.array([sv[0,j], 0.]))
		e = np.sqrt(sv)
		# semimajor axis
		a = L**2 / (G * mass[n_gal*N//2] * (1-e**2))
		# period in TU
		P = 2 * np.pi * np.sqrt(a**3 / (G * mass[n_gal*N//2]))
		# frequency in 1/TU
		f[n_gal*N//2:(n_gal+1)*N//2] = 1/P

	# rescale to middle C (256 Hz)
	fmed = np.nanmedian(f)
	f = f * 256/fmed
	
	return virnorm, f

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

@nb.njit
def get_sigma_ij( vel ):

	s = np.zeros((3, 3))
	for i in nb.prange(3):
		for j in nb.prange(3):
			vi_avg = np.nanmean(vel[:, i])
			vj_avg = np.nanmean(vel[:, j])
			s[i, j] = np.nanmean(np.sqrt((vel[:, i] - vi_avg) * (vel[:, j] - vj_avg)))
	
	return s

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1].astype(int), x[1:].astype(int), out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def main():
	""" N-body simulation """
	
	# Simulation parameters
	N         = 242    # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 100.0   # time at which simulation ends
	dt        = 0.01   # timestep
	softening = 0.0    # softening length TURN THIS OFF
	G         = 1.0    # Newton's Gravitational Constant
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	pos = np.zeros((N,3))
	vel = np.zeros((N,3))
	mass = np.ones((N,1)) / N / 100

	# mass[0] = 1
	# mass[1] = 1e-4

	# pos[0,:] = (0,0,0)
	# vel[0,:] = (0,0,0)

	# pos[1,:] = (2,0,0)
	# vel[1,:] = (0,1/np.sqrt(4)+0.05,0)

	r_scale = 6        # radial separation of each ring

	# Central, massive, at rest particles
	mass[0] = 1
	pos[0,:] = (0,0,0)
	vel[0,:] = (0,0,0)

	mass[N//2] = 1
	pos[N//2,:] = (5*r_scale,0,0)
	r_m = 1/np.sqrt(2*r_scale)+0.05
	vel[N//2,:] = (-r_m * 0.8,r_m * 0.5,0)

	inc = (0, np.pi/4)       # inclination of the disk orientation, 0 = face-on in the x-y plane, pi/2 = edge-on

	# n_ring = 6 * np.arange(2, 14, 1)
	n_ring = 6 * np.arange(2, 7, 1)
	r_percent = np.linspace(0.2, 0.6, len(n_ring))

	for n_gal in range(2):
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
			v_ring = np.sqrt(G * mass[0] / r_ring)
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
	acc = getAcc( pos, mass, G, softening )
	
	# calculate initial energy of ONLY galaxy 1
	virnorm, freq  = getEnergy( pos, vel, mass, G )

	# rough estimate of free-fall time
	t_ff = estimate_freefall( N, pos, vel, mass, G )

	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))

	# velocity dispersion tensor
	sigma = get_sigma_ij( vel )
	# Save sigma for each time step
	sigma_sav = np.zeros((3,3,Nt+1))
	sigma_sav[:,:,0] = sigma
	
	# save energies, particle orbits for plotting trails
	pos_save = np.zeros((N,3,Nt+1))
	pos_save[:,:,0] = pos
	vir_save = np.zeros(Nt+1)
	vir_save[0] = virnorm
	t_all = np.arange(Nt+1)*dt

	# make ACTUAL matplotlib animations
	# prep figure
	fig = plt.figure(figsize=(10,7), dpi=80)
	grid = plt.GridSpec(6, 6, wspace=1.0, hspace=0.0)
	ax1 = plt.subplot(grid[0:5,0:3])
	ax2 = plt.subplot(grid[5,:])
	ax3 = plt.subplot(grid[0:5,3:6])

	xx, yy = pos_save[:,0,0], pos_save[:,1,0]
	sca = ax1.scatter([], [], color=[.7,.7,1], s=1)
	sca0 = ax1.scatter(xx, yy, color='blue', s=2)
	eng = ax2.scatter([], [], label='$t_{ff} \sim %.2f$' % t_ff, color='red', s=1)

	counts, edges = np.histogram(freq, bins=np.arange(0, 1050, 50))
	fre = ax3.stairs(counts / np.max(counts), edges, color='blue', lw=1)

	ax1.set(xlim=(-2*r_scale, 2*r_scale), ylim=(-2*r_scale, 2*r_scale))
	ax1.set_aspect('equal', 'box')
	ax1.set_xlabel('$x$')
	ax1.set_ylabel('$y$')
	# ax1.set_xticks(np.arange(-12, 16, 4))
	# ax1.set_yticks(np.arange(-12, 16, 4))

	ax2.set(xlim=(0, tEnd), ylim=(-3, 3))
	# ax2.set_aspect(tEnd/8, 'box')

	ax2.set_xlabel('time')
	ax2.set_ylabel('virial')
	# ax2.legend(loc='upper right')

	ax3.set_xlabel('freq (Hz)')
	ax3.set_ylabel('number (normalized)')
	ax3.set_aspect(1000, 'box')
	# ax3.set_xscale('log')
	# ax3.set_yscale('log')
	ax3.set_xlim(0, 1000)
	ax3.set_ylim(0, 1)

	FFMpegWriter = ani.writers['ffmpeg']
	writer = FFMpegWriter(fps=30)

	mf = MIDIFile(1, deinterleave=False)
	track = 0
	mf.addTrackName(track, 0, "galaxy")
	mf.addTempo(track, 0, 100/(Nt/5/30/60))

	mf.addProgramChange(track, 0, time=0, program=40)  # Channel 0: Violin, particle ring 1
	mf.addProgramChange(track, 1, time=0, program=6)   # Channel 1: Harpsichord, particle ring 2 
	mf.addProgramChange(track, 2, time=0, program=40)  # Channel 2: Violin, particle ring 3
	mf.addProgramChange(track, 3, time=0, program=6)   # Channel 3: Harpsichord, particle ring 4
	mf.addProgramChange(track, 4, time=0, program=42)  # Channel 4: Cello, particle ring 5
	# N.B.: Channel 9 is drums

	pitches = np.zeros((Nt, N-2), dtype=np.int16)

	with writer.saving(fig, 'nbody.mp4', 100):
		
		# Simulation Main Loop
		for i in tqdm.trange(Nt):
			# (1/2) kick
			vel += acc * dt/2.0
			
			# drift
			pos += vel * dt
			
			# update accelerations
			acc = getAcc( pos, mass, G, softening )

			sigma = get_sigma_ij( vel )
			sigma_sav[:,:,i+1] = sigma
			
			# (1/2) kick
			vel += acc * dt/2.0
			
			# update time
			t += dt
			
			# get energy of only galaxy 1
			virnorm, freq = getEnergy( pos, vel, mass, G )
			
			# save energies, positions for plotting trail
			pos_save[:,:,i+1] = pos
			vir_save[i+1] = virnorm

			# only save every X frames (for efficiency)
			if i % 5 == 0:
				# plot in real time
				xx = pos_save[:,0,max(i-50,0):i+1] - pos_save[0,0,max(i-50,0):i+1]
				yy = pos_save[:,1,max(i-50,0):i+1] - pos_save[0,1,max(i-50,0):i+1]
				sca.set_offsets(np.c_[xx.ravel(), yy.ravel()])
				sca0.set_offsets(np.c_[pos[:,0]-pos[0,0], pos[:,1]-pos[0,1]])
				eng.set_offsets(np.c_[t_all[vir_save != 0],vir_save[vir_save != 0]])

				counts, edges = np.histogram(freq, bins=np.arange(0, 1050, 50))
				fre.set_data(counts / np.max(counts), edges)
				
				writer.grab_frame()
			
			for n_gal in range(2):
				low = n_gal*N//2
				upp = (n_gal+1)*N//2
				# Convert frequency to MIDI pitch
				pitches[i, low:upp-2] = (12*np.log(freq[low+1:upp-1]/440)/np.log(2) + 69).astype(int)

			# save every 1/10th of simulation
			if i % (Nt//10) == 0 or i == (Nt-1):
				plt.savefig(f"nbody_{i}.png", dpi=300, bbox_inches='tight')
	
	plt.close()

	for ni in range(len(n_ring)):

		# lower index on matrices based on previous n_rings, offset by 1 because of central mass
		low = 1 + int(np.sum(n_ring[0:ni]))
		# upper index '' '' ''
		upp = 1 + int(np.sum(n_ring[0:ni+1]))

		run_vals, run_starts, run_lens = find_runs(np.nanmean(pitches[:, low:upp], axis=1))
		for rs, rv, rl in zip(run_starts, run_vals, run_lens):
			while rv > 255:
				rv -= 12
			while rv < 0:
				rv += 12
			if ni in (0, 2):
				# Go up an octave for violin and viola
				rv += 12
			if ni in (3, 4):
				# Go down an octave for cello and one harpsichord
				rv -= 12
			volume = (upp - low) * 100 / np.max(n_ring)
			mf.addNote(track, ni, pitch=int(rv), time=rs/100, duration=rl/100, volume=80)
	
	# Drum beat corresponds to dynamical "hotness"
	# velocity dispersion tensor
	# hotness = np.nansum(sigma_sav, axis=(0, 1))
	# hotness = hotness / np.nanmax(hotness) * 0.0625 # max hotness corresponds to 16th notes
	# td = 0
	# for hi in hotness:
	# 	n_notes = int(np.ceil(1/hi))
	# 	for n in range(n_notes):
	# 		mf.addNote(track, 9, pitch=37, time=td/100, duration=hi/100, volume=110)
	# 		td += hi/100

	with open("galaxy.mid", "wb") as file:
		mf.writeFile(file)
	
	# FluidSynth().midi_to_audio("galaxy.mid", "galaxy.mp3")
	cmd = 'timidity galaxy.mid -Ow'
	subprocess.call(cmd, shell=True)
	cmd = 'ffmpeg -y -i galaxy.wav  -r 30 -i nbody.mp4  -filter:a aresample=async=1 -c:a flac -c:v copy galaxy.mkv'
	subprocess.call(cmd, shell=True)

	return 0


  
if __name__== "__main__":
  main()
