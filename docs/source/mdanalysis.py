#!/usr/bin/env python
# coding: utf-8

# # Comparison with MDAnalysis
# 
# `MDAnalysis` also contains a tool for calculating the mean-squared displacements. So why use `kinisi` over `MDAnalysis`?
# 
# Well, the approach taken by `kinisi`, which is outlined in the [methodology](./methodology.html), uses a high precision approach to estimate the diffusion coefficent and offers an accurate estimate of the variance in the mean-squared displacements and diffusion coefficient from a single simulation.
# 
# In this notebook, we will compare the results from `MDAnalysis` and `kinisi`. 
# First we will import the `kinisi.analyze.DiffusionAnalyzer` and `MDAnalysis.analysis.msd` classes.

# In[ ]:


import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import scipp as sc

import kinisi
from kinisi.analyze import DiffusionAnalyzer
import MDAnalysis.analysis.msd as msd
from MDAnalysis.transformations.nojump import NoJump


# Next, we are going to need to do a little 'magic' to get `MDAnalysis` to read an `extended xyz` file.
# Luckly `ASE` can help us out.

# In[ ]:


from ase.io import read
from MDAnalysis import Universe
import os


# Since this trajectory has a triclinic cell; we need to extract the cell dimensions and pass them to `MDAnalysis`. 

# In[ ]:


# The file we want to load is in the test cases for kinisi, so will need to get the path to it.
path_to_file = os.path.join(os.path.dirname(kinisi.__file__), 'tests/inputs/LiPS.exyz')
atoms = read(path_to_file, format='extxyz', index=':')

cell_dimensions = []
for frame in atoms:
    lengths = frame.cell.lengths()
    angles = frame.cell.angles()
    cell = [*lengths, *angles]
    cell_dimensions.append(cell)


# We now have the cell dimensions ($a, b, c, \alpha, \beta, \gamma$) in an array. We now need to create an `MDAnalysis` `universe` and apply these dimension to it.

# In[ ]:


u = Universe(path_to_file, path_to_file, format='XYZ', topology_format='XYZ', dt=20.0/1000)
for ts, dims in zip(u.trajectory, cell_dimensions):
    ts.dimensions = dims


# Now we add an unwrapping transformation to the `universe`, create the `MSD` object, run the analysis, and extract the results as a timeseries.

# In[ ]:


u.trajectory.add_transformations(NoJump())
MSD = msd.EinsteinMSD(u, select='type LI', msd_type='xyz', fft=True, verbose=False)
MSD.run(verbose=False)
mda_MSD = MSD.results.timeseries
mda_dt = np.linspace(u.trajectory.dt, u.trajectory.dt * len(mda_MSD), len(mda_MSD))


# With the result from `MDAnalysis` in one hand we can now do the same thing in `kinisi` with the other.
# Here, we use the custom `dt` functionality that is available in `kinisi`, with this we pass a `scipp` array of time intervals to compute the MSD over.
# Note, that the custom `dt` must be represented within possible simulation time intervals, or an error will be raised.  

# In[ ]:


params = {'specie': 'LI',
          'time_step': 0.001 * sc.Unit('ps'),
          'step_skip': 20 * sc.units.dimensionless,
          'progress': False,
          'dt': sc.arange(dim='time interval', start=0.1 * sc.Unit('ps'),
                         stop=4.1 * sc.Unit('ps'),
                         step=0.1 * sc.Unit('ps'))
          }

kinisi_from_universe = DiffusionAnalyzer.from_universe(u, **params)
kinisi_MSD = kinisi_from_universe.msd
time = kinisi_from_universe.dt


# Putting the two analyses' together, we can plot for comparison.

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(time, kinisi_MSD, label='Kinisi MSD', c='#ff7f0e')
plt.plot(mda_dt, mda_MSD, label='MDanalysis', ls='--')
plt.ylabel('MSD (Å$^2$)')
plt.xlabel('Diffusion time (ps)')
plt.legend()


plt.show()


# The results overlap almost entirely.
# 
# We can now extract an accurate estimation of the variance in the observed MSD from `kinisi`. For clarity, we will use array indexing to remove every other data point.

# In[ ]:


plt.errorbar(kinisi_from_universe.dt.values,
             kinisi_from_universe.msd.values,
             np.sqrt(kinisi_from_universe.msd.variances),
             c='#ff7f0e')
plt.ylabel(r'MSD/Å$^2$')
plt.xlabel(r'$\Delta t$/ps')
plt.show()


# We can also calculate estimated diffusion coefficient, and the associated uncertainty, with `kinisi`.

# In[ ]:


kinisi_from_universe.diffusion(1.5 * sc.Unit('ps'))
kinisi_from_universe.D


# In[ ]:


plt.hist(kinisi_from_universe.D.values)


# In[ ]:


credible_intervals = [[16, 84], [2.5, 97.5], [0.15, 99.85]]
alpha = [0.6, 0.4, 0.2]

plt.plot(kinisi_from_universe.dt, kinisi_from_universe.msd, 'k-')
for i, ci in enumerate(credible_intervals):
    plt.fill_between(kinisi_from_universe.dt.values,
                     *np.percentile(kinisi_from_universe.distributions, ci, axis=1),
                     alpha=alpha[i],
                     color='#0173B2',
                     lw=0)
plt.ylabel('MSD/Å$^2$')
plt.xlabel(r'$\Delta t$/ps')
plt.show()


# In[ ]:


kinisi_from_universe.da

