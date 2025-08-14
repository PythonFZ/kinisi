# Uncertainty quantification in diffusion

[![JOSS Status](https://joss.theoj.org/papers/1ae102ffb6b3c63b04c002976440815d/status.svg)](https://joss.theoj.org/papers/1ae102ffb6b3c63b04c002976440815d)
[![Test Coverage](https://api.codeclimate.com/v1/badges/3e64239fb6cb6c837b62/test_coverage)](https://codeclimate.com/github/bjmorgan/kinisi/test_coverage)
[![Documentation Status](https://readthedocs.org/projects/kinisi/badge/?version=latest)](https://kinisi.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/kinisi.svg)](https://badge.fury.io/py/kinisi)

`kinisi` is an open-source Python package that can accurately estimate diffusion processes in atomic and molecular systems and determine an accurate estimate of the uncertainty in these processes.

This is achieved by modelling the diffusion process as a [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution), based on that for a random walker. 
This ensures and accurate estimation of the diffusion coefficient and it's uncertainty.
More information about the approach `kinisi` uses can be found in the [methodology article](https://doi.org/10.1021/acs.jctc.4c01249), which is also introduced in this [poster](./_static/poster.pdf).

`kinisi` can handle simulation trajectories from many common molecular dynamics packages, including [VASP](https://www.vasp.at/wiki/index.php/XDATCAR) and those that can be read by [MDAnalysis](https://userguide.mdanalysis.org/stable/reading_and_writing.html) and [ASE](https://wiki.fysik.dtu.dk/ase/ase/io/io.html).
Examples of some of these analyses are shown in the [notebooks](./notebooks), also given there is an example of using `kinisi` to investigate the Arrhenius relationship of diffusion as a function of temperature.

```{image} ./_static/example_light.png
  :width: 450
  :align: center
  :class: only-light
  :alt: An example of the kinisi analysis for the diffusion of lithium in a superionic material. 
```
```{image} ./_static/example_dark.png
  :width: 450
  :align: center
  :class: only-dark
  :alt: An example of the kinisi analysis for the diffusion of lithium in a superionic material. 
```
<center>
<small>
An example of the output from <code>kinisi</code>; showing the determined mean-squared displacements (solid line),<br>
the estimated Einstein diffusion relationship (blue regions representing descreasing credible intervals).
</small>
</center>

## Contributors

[Andrew R. McCluskey](https://www.mccluskey.scot) | [Benjamin J. Morgan](https://morgan-group-bath.github.io) | [Alex G. Squires](https://alexsquires.github.io) | Josh Dunn

```{toctree}
---
hidden: True
---

installation
methodology
notebooks
faq
papers
modules
```
