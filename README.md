# PoSI-GroupLASSO
Post-selection Inference for Group Lasso Penalized M-Estimators

## Installation
    virtualenv env3 -p python3
    source env3/bin/activate
    pip install -r requirements.txt
To install the package `regreg`, do:

    pip install git+https://github.com/regreg/regreg.git

## Potential Issues & Solutions
1. We used the Poisson regression functionality in `regreg` to solve for certain parameters
in simulations for Poisson regression, the original `regreg` codes require integer responses,
which may not be the case in our simulation setup. 
In these occasions, the original `regreg` file will produce an error message prompting for an
integer response and stop running, which can be solved by commenting out lines 1048 and 1049
in env3/lib/python3.10/site-packages/regreg/smooth/glm.py
