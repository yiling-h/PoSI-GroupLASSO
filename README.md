# PoSI-GroupLASSO
Post-selection Inference for Group Lasso Penalized M-Estimators

## Cloning
```angular2html
   git clone https://github.com/yiling-h/PoSI-GroupLASSO.git
    # then cd into ./PoSI-GroupLASSO
   git switch pysi
```

## Installation
```angular2html
    virtualenv env3 -p python3 
    #(alternatively virtualenv env3 -p python3.10)
    source env3/bin/activate
    pip install -r requirements.txt
```
To install the package `regreg`, do:

    pip install git+https://github.com/regreg/regreg.git

## Jupyter Notebook
To use the code in Jupyter Notebook, do:
```angular2html
    pip install -e .
    pip install jupyter ipykernel
    python -m ipykernel install --user --name env3 --display-name "Python (env3)"
```
Then run the code using the kernel "Python (env3)".

A more detailed tutorial of installation can be found at `selectinf/Replicability/replication_tutorial.ipynb` 
in this repository.

## Potential Issues & Solutions
1. We used the Poisson regression functionality in `regreg` to solve for certain parameters
in simulations for Poisson regression, the original `regreg` codes require integer responses,
which may not be the case in our simulation setup. 
In these occasions, the original `regreg` file will produce an error message prompting for an
integer response and stop running, which can be solved by commenting out lines 1048 and 1049
in env3/lib/python3.10/site-packages/regreg/smooth/glm.py
2. Due to package compatibility, a python version >= 3.10 is recommended for the virtual environment.

## Related Paper \& Replicability
1. The corresponding paper with theoretical results can be found at [link to paper](https://arxiv.org/pdf/2306.13829.pdf).
2. To replicate the simulation section, one can refer to the following files under the path `selectinf/Simulation`:
   1. Gaussian link: `gaussian_simulation.py`
   2. Logistic link: `logistic_simulation.py`
   3. Poisson link: `logistic_simulation.py`
   4. Quasi-Poisson modeling for negative binomial responses: `quasipoisson_simulation.py`
3. An interactive Jupyter Notebook that runs small scale replications of all four experiments is given at
`selectinf/Replicability/replication_tutorial.ipynb`. 
To run the Jupyter Notebook using the virtual environment `env3` created earlier, 
it is recommended to open the project using a Python IDE such as PyCharm.

## Troubleshooting
For potential issues and mistakes, please contact Yiling Huang (yilingh@umich.edu) for correction.
