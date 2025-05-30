## stability_sGMRES

This code can be used to reproduce the results of the numerical experiments presented in

[1] L. Burke and E. Carson, and Y. Ma, "On the numerical stability of sketched GMRES", arxiv:2503.19086, 2025

Our code uses (with modifications) some of the code provided with the paper

[2] S. Güttel and I. Simunec, "A sketch-and-select Arnoldi process", SIAM Journal on Scientific Computing 46.4 (2024)

available at https://github.com/simunec/sketch-select-arnoldi. 

## Included Tests
**test_random.m**: Generates Figure 1.

**test_torso.m**: Generates Figure 2.

**test_stomach.m**: Generates Figure 3.

**test_adapt_restart_stomach.m**:  Generates Figure 4 a.

**test_adapt_restart_fs_760_1.m**: Generates Figure 4 b.

**test_adapt_restart_shermann.m**: Generates Figure 4 c.

## Code Requirements
The code has been tested and developed using MATLAB 2023a.

The script **test_torso.m** requires the torso3 matrix (ID: 898) from [SuiteSparse](https://sparse.tamu.edu/).
