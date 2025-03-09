# Scientific Computing Assignment 2 (SCICOMP_22)

## Project Overview
This project explores different numerical methods for modeling and simulating diffusion-limited aggregation (DLA) and reaction-diffusion systems. The project is structured into three parts:

1. **Diffusion Limited Aggregation (DLA) Discretization**: Using a discretized probability-based growth model to approximate DLA growth patterns.
2. **Monte Carlo Simulation of DLA**: Simulating DLA using a random walker approach to validate the discretized solution.
3. **Gray-Scott Reaction-Diffusion Model**: Solving the reaction-diffusion equations numerically and visualizing the effects of control parameters on pattern formation.

These models are used to study pattern formation and growth dynamics in complex systems. The project includes implementations in Python with visualization and performance analysis.

## Repository Structure
```
SCICOMP_22/
├── fig/                           # Directory containing result images
├── src/                           # Source code directory
│   │ 
│   ├── matijs/                    # Part I:   DLA Discretization solution
│   │   ├── DLA_final.py           # DLA sequential and optional parallel implementation
│   │ 
│   ├── noa/                       # Part II:  Random walker for DLA
│   │   ├── monte_carlo_DLA.py     # core part of Monte Carlo simulation for DLA
│   │   ├── metrics.py             # metrics, used for comparing the random walker with the Discretization solution in part I
│   │ 
│   ├── alex/                      # Part III: Gray-Scott model
│   │   ├── reaction_diffusion.py  # reaction-diffusion discretization implementation, and ploting/animation functions
│   
├── .gitignore                     # Git ignore file
├── LICENSE                        # License information
├── presentation.ipynb             # Jupyter Notebook for presentation
├── requirements.txt               # Dependencies list for Python installation
├── README.md                      # Project documentation
```

## Description of Files
### Source Code (`src/`)
- **matijs/**: Contains the discretized solution for Diffusion Limited Aggregation (DLA), including probabilistic growth modeling.
- **noa/**: Implements the Monte Carlo simulation for DLA using random walkers and includes comparison metrics.
- **alex/**: Includes the numerical solution for the Gray-Scott reaction-diffusion system, along with visualization scripts.

### Figures (`fig/`)
- This directory contains all result images generated from the computations.

### Other Files
- **LICENSE**: License file for the repository.
- **README.md**: This document, providing an overview of the project.

## Running the Code
To execute the different parts of the assignment, navigate to the **presentation.ipynb**, which contains overview and visualizations.

## Dependencies
Ensure the following Python packages are installed:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the terms specified in the `LICENSE` file.

## Contact
For any questions, please refer to the presentation notebook or the provided scripts.
