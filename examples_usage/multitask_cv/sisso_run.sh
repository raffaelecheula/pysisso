#!/bin/bash

module load intel/2020.4
module load openmpi/4.0.3
ulimit -s unlimited

python sisso_regression.py
