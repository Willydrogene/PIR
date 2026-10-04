# Tidal Inertial Waves in Compact Exoplanetary Systems

Research initiation project carried out at **IRAP-CNRS** as part of the
MSc in Astrophysics, Space Science and Planetology (ASEP) at the
University of Toulouse.

**Supervisor:** Aurélie Astoul

**Students:** [@Willydrogene](https://github.com/Willydrogene),
[@Zoltrak-Kiruwa](https://github.com/Zoltrak-Kiruwa) and
[@Mannu-aile](https://github.com/Mannu-aile)

## Project Overview

This project investigates nonlinear tidal inertial waves in rotating stars
and giant planets using data from hydrodynamical simulations produced with
the pseudo-spectral MagIC code.

We analysed more than 70 GB of simulation data and studied the evolution of
spherical-harmonic modes in order to identify emerging non-harmonic
frequencies and investigate possible signatures of nonlinear interactions.

The project involved:

- analysis of hydrodynamical simulation data;
- spectral analysis of spherical-harmonic modes;
- investigation of the limitations of FFT methods for irregularly sampled
  time series;
- implementation of Lomb-Scargle periodograms using Astropy;
- automatic detection of significant emerging frequencies;
- investigation of possible triadic resonances and subharmonic instabilities;
- study of the influence of viscosity on the nonlinear tidal response.

## Report

The final written report is available here:

**[PIR_report.pdf](./PIR_report.pdf)**

The report provides the most complete description of the scientific context,
methods, analysis and results of the project.

## Oral Presentation

The slides used for the final oral presentation are also available:

**[PIR_Oral.pdf](./PIR_Oral.pdf)**

They provide a shorter overview of the project and its main results.

## Code

This repository contains part of the Python code developed and used during
the project. Different stages of the work can be found across the repository
branches.

The repository should **not be considered a complete archive of the final
code**, as some scripts and working files remained local during the project.

For a complete and structured presentation of the work, please refer primarily
to the **final report** and the **oral presentation**.

## Technologies and Methods

- **Python** — data processing and analysis
- **NumPy** — numerical data manipulation
- **Astropy** — Lomb-Scargle periodograms
- **Matplotlib** — scientific visualisation
- **FFT and spectral analysis**
- **Spherical-harmonic mode analysis**

## Institution

**IRAP-CNRS — Institut de Recherche en Astrophysique et Planétologie**  
Université de Toulouse, France
