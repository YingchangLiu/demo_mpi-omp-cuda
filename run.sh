#!/bin/bash

 
make clean
make all

mpirun -np 8 ./mymoc