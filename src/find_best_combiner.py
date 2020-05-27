#!/usr/bin/env python3
"""
[TASK 4]:
  This code help to find an improved combiner function (less correlation), than the one in Geffe's generator
  The program aims to find the correlation immunity of order one and two, non linear order and balancedness.
"""
import math

"""
Change accordingly:
"""
BALANCED = False
CORR_RES_ORD_ONE = 2
CORR_RES_ORD_TWO = 1
NON_LIN_ORD = 2

"""
Functions
"""


def find(s, ch):
  return [x for x, ltr in enumerate(s) if ltr == ch]


def get_multiple_characters(s, indexes):
  for i in indexes:
    if s[i] != "0":
      return None
  return s


def get_ANF(f):
  varnum = int(math.log2(len(f)))
  n_rows = int("0b" + "1" * varnum, 2)
  sequence = [(bin(i)[2:].zfill(varnum), gc) for i, gc in zip(range(n_rows + 1), f)]
  # The Möbius transform
  endsumvals = []
  for i in range(n_rows + 1):
    a = []
    u = bin(i)[2:].zfill(varnum)
    for x in sequence:
      for y in sequence:
        if y[0] == get_multiple_characters(x[0], find(u, "0")):
          a.append(int(y[1]))
    if sum(a) % 2 == 1:
      endsumvals.append(u)
  return endsumvals


def get_correlation(x, f):
  q = [0, 0, 0]
  for x, f in zip(x, f):
    for i in range(3):
      if x[i] == f:
        q[i] += 1

  return [i / 8 for i in q]


"""
Main
"""

x_vals = []
for i in range(8):
  x_vals.append(list(map(int, bin(i)[2:].zfill(3))))

x_nlot = [[x[0] ^ x[1], x[0] ^ x[2], x[1] ^ x[2]] for x in x_vals]

for f_int in range(1, int("1" * 8, 2)):
  bin_f = list(map(int, bin(f_int)[2:].zfill(8)))
  if sum(bin_f) != 4 and BALANCED:  # balanced
    continue
  else:
    # Correlation order one and two
    q1 = get_correlation(x_vals, bin_f)
    q2 = get_correlation(x_nlot, bin_f)
    # balanced-count
    bc = bin_f.count(1)
    # ANF
    anf = get_ANF(''.join(map(str, bin_f)))
    nlinc = max([ai.count('1') for ai in anf])
    # Adjust this according to the needs:
    if q1.count(0.5) >= CORR_RES_ORD_ONE and q2.count(0.5) >= CORR_RES_ORD_TWO and nlinc == NON_LIN_ORD:
      print(f"f(x):{bin_f} --> {bc}\n\tcorrelation immune order 1:{q1}\n\tcorrelation immune order 2:{q2}")
      print(f"\tNon linear order: {nlinc} --> {anf}\n")
