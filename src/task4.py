#!/usr/bin/env python3

# GLOBAL VARIABLES
INFILE = "plaintextfiles/1000_example.txt"
Z1S = 123
Z2S = 90
Z3S = 577

from lib.lfsr import lfsr


def task4(seeds):
  print("[TASK1]: Initial setup of polynomials..")
  z1 = lfsr([10, 7, 3, 1])
  z2 = lfsr([20, 15, 12, 8, 6, 5])
  z3 = lfsr([11, 7, 3, 1])

  # set the key / seed values for the LFSR's (needs to be less than 2^(length of LFSR)
  # THis is the key!:
  z1.set_seed(seeds[0])
  z2.set_seed(seeds[1])
  z3.set_seed(seeds[2])


if __name__ == "__main__":
  print(" TASK 4".center(30, "#"))
  task4([Z1S, Z2S, Z3S])  # Improved Geffe's generator
