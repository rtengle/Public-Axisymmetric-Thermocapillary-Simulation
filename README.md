# Axisymmetric Simulation of Curved Liquid Film

This is the companion code for:

> **R. Engle and V. Frumkin**, "Thermocapillary Stabilization of Liquid Space Telesclopes" (2026).

This repository contains the code utilized to simulate the combined surface-temperature liquid mirror dynamics. The code utilizes backwards Euler scheme to simulate each time step and a 9-point finite difference method to calculate the system derivatives. 

## System Overview

The code implements the long-wave model given in Equations 2.16, 2.21, 2.22, and 2.24 with the original polar angle and the addition of a transverse conduction term to Equation 2.16:

$$\begin{gather} 
\frac{\partial H}{\partial \tau} + M \nabla_2 \cdot \left(\frac{1}{3} S H^3 \nabla_2 (2 H + \nabla_2^2 H) - \frac{1}{2}H^2 \nabla_2 \theta \Big|_{R=-H} \right) = 0 
\\
\frac{\partial \theta}{\partial \tau} + M \left(U_r \frac{\partial \theta}{\partial R} + U_\gamma \frac{\partial \theta}{\partial \gamma}  \right) = \frac{\partial^2 \theta}{\partial R^2} + \epsilon^2 \nabla_2^2 \theta
\\
U_r = \nabla_2 \cdot \left(\frac{1}{3} S \left(\frac{1}{6}R^2 + \frac{1}{2}HR\right) \nabla_2 (2 H + \nabla_2^2 H) - \frac{1}{2}R^2 \nabla_2 \theta \Big|_{R=-H} \right)
\\
U_\gamma = -\frac{1}{3} S \left(\frac{1}{2}R^2 + HR \right)\nabla_2 (2 H + \nabla_2^2 H) + R \nabla_2 \theta \Big|_{R=-H}
\end{gather}$$

The inner boundary ($\gamma = 0$) follows the axisymmetric condition, where all odd derivatives are zero. The outer boundary ($\gamma = \gamma_0$) follows a pinned condition from the set surface and assumes to additional curvature in the surface and temperature:

$$\begin{gather} 
\frac{\partial H}{\partial \gamma} \Big|_{\gamma=0} = \frac{\partial^3 H}{\partial \gamma^3} \Big|_{\gamma=0} = \frac{\partial \theta}{\partial \gamma} \Big|_{\gamma=0} = 0
\\
H\Big|_{\gamma=\gamma_0} = H_{init}\Big|_{\gamma=\gamma_0}, \quad \nabla_2^2 H\Big|_{\gamma=\gamma_0} = \nabla_2^2 \theta\Big|_{\gamma=\gamma_0}  = 0
\end{gather}$$

## Repository Structure

```
FLUTE-liquid-mirror-dynamics/
├── src/                                Simulation functions
│   ├── FiniteDifference.py             Automatic calculation of finite difference coefficients
│   ├── LegendreMode.py                 Calculating the Legendre function and finding specific modes
│   ├── ParallelProcessing.py           TestSeries class that runs multiple tests at once
│   ├── SurfaceSystem.py                Surface class containing surface dynamics of Eq. 2.24
│   ├── TemperatureSystem.py            Temperature class containg thermodynamics of Eq. 2.16, 2.21, and 2.24
│   ├── ThinFilmSystem.py               ThinFilm class containing solver loop and recording process.
│   └── Units.py                        Units class that converts dimensional quanitites to revelant parameters.
│
├── Mode Sweep.py                       Simulation script to test modes 1, 2, 3, and 5 concurrently.
├── Mode Comparison.py                  Plotting script to compare results to linear solution.                
├── LICENSE
├── .gitignore
└── README.md
```



