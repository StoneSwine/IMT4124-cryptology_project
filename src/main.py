#!/usr/bin/env python3
import math  # math innit
import os  # get full filepath

import matplotlib.pyplot as plt  # Create graphs
import numpy as np
from columnar import columnar  # Pretty print tables
from scipy.stats import norm  # norm.cdf(1.96) and norm.ppf(norm.cdf(1.96))

"""
LFSR CLASS
"""


# initialized with the polynomial, and the seed are added dynamically
class lfsr(list):
  """initialized with the polynomial, initial state (seed) are added dynamically
  """

  def __init__(self, l):
    super().__init__(l)
    self.degree = max(l)

  def get_polynomial(self):
    """
    Prettyprint the polynomial
    :return: The polynomial in a nice format
    """
    str = "1"
    for i in reversed(self):
      str += f"+x^{i}"
    return str

  def get_current_state(self):
    """
    :return: he current value of the register in int
    """
    return self.reg

  def get_seed(self):
    return self.start_val

  def set_seed(self, seed):
    self.reg = self.start_val = seed

  def next_o(self):
    """
    Shift the register one bit
    :return: The output bit from that shift
    """
    out = self.reg & 1  # get the LS bit
    b = 0
    for p in self:  # loop the polynomial degrees (index of bits to xor)
      p = self.degree + 1 - p
      b ^= ((self.reg >> p - 1) & 1)  # shift the register accordingly and get the LSbit to XOR with the other ones
    self.reg = (self.reg >> 1 | b << (self.degree - 1))  # Shift the register and apply the new bit (will also pad)
    return out

  def get_degree(self):
    """
    :return: The degree of the LFSR
    """
    return self.degree

  def get_x_rounds(self, x=1):
    """
    :param x: Number of rounds to run (default 1)
    :return: The output from the given rounds
    """
    o_a = []
    for _ in range(x):
      o_a.append(self.next_o())
    return o_a

  def run_period(self):
    """
    :return: The output from the whole period of the LFSR
    """
    ret = []
    while True:
      ret.append(self.next_o())
      if self.reg == self.start_val:
        return ret


"""
GLOBAL VARIABLES
"""
INFILE = "plaintextfiles/4000_example.txt"
Z1S = 69
Z2S = 190
Z3S = 574

DEMO = True  # A little cheat to make the bruteforce exit when the right seed is found
INCLUDE_STATISTICS = False  # This takes quite a bit of time

z1 = lfsr([10, 7, 3, 1])
z2 = lfsr([20, 15, 12, 8, 6, 5])
z3 = lfsr([11, 7, 3, 1])

"""
HELPER FUNCTIONS
"""


def gg_combining_function(z1, z2, z3):
  """
  Geffes generator combining function
  :param z1: Output bit from L1
  :param z2: Output bit from L2
  :param z3: Output bit from L3
  :return: Output X from Geffes generator
  """
  return (z3 ^ (z1 & z2) ^ (z2 & z3))


def run_correlation_attack(qi, p0, c, z, pf=0.002):
  """
  Perform correlation attack on a LFSR
  :param qi: The correlation from truth table between the register and the output from geffes generator
  :param p0: Probability of the input laguage
  :param c: the ciphertext in array of bits
  :param z: an instance of "lfsr" for the register to attack
  :param pf: probability of false positives
  :return: possible candidate initial states
  """
  candidates = []
  pe = 1 - (p0 + qi) + 2 * p0 * qi
  l = len(c)  # This can be varied, depending on how much you are reading, and will influence the rest
  T = norm.ppf(1 - pf) * math.sqrt(l)
  #                                                 standardise variables
  #    1 -  the cumulative density function of (mean / Standard Deviation (square of variance)) =  (survival function)
  pm = 1 - norm.cdf((l * (2 * pe - 1) - T) / (math.sqrt(4 * l * pe * (1 - pe))))
  Ri = z.get_degree()
  print("[TASK3]: Information about variables:")
  print(f"\tpe: {pe}\n\tpm: {pm}\n\tpf: {pf}\n\tl: {l} bit\n\tT: {T:.1f}\n\tRi: {Ri}")
  print("[TASK3]: Seed | Alpha")
  z.set_seed(1)
  period = np.array(z.run_period())
  z.set_seed(1)
  for i in range(len(period)):
    alpha = l - (2 * sum([ci ^ zi for ci, zi in zip(c, period.take(range(i, i + l), mode="wrap"))]))
    if alpha >= T:
      print(f"[TASK3]: {z.get_current_state()}  | {alpha}")
      candidates.append(z.get_current_state())
    z.next_o()
  return candidates


def bitgen(x):
  """
  Generate bits from an input
  :param x: bytes
  :return: Generator object
  """
  for c in x:
    for i in range(8):
      yield int((c & (0x80 >> i)) != 0)


def bits2string(bits=None):
  """
  This function is taken from https://stackoverflow.com/a/10238140
  :param bits: Array of bits
  :return: String
  """
  chars = []
  for b in range(len(bits) // 8):
    byte = bits[b * 8:(b + 1) * 8]
    chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
  return ''.join(chars)


def entropy(string):
  """
  This function is taken from https://stackoverflow.com/a/2979208
  get probability of chars in string and calculate the entrophy
  :param string:
  :return: entrophy in float
  """
  prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
  entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
  return entropy


"""
THE MAIN TASKS
"""


def task1(plaintextfile, seeds):
  """
  TASK 1: Generate ciphertext from input file, using geffes generator as the pseudorandom function / input
  :param plaintextfile: filename in string
  :param seeds: array of seeds for the three registers
  :return: ciphertext in array of bits
  """
  global z1, z2, z3
  print("[TASK1]: Initial setup of polynomials..")

  # set the key / seed values for the LFSR's (needs to be less than 2^(length of LFSR)
  # THis is the key!:
  z1.set_seed(seeds[0])
  z2.set_seed(seeds[1])
  z3.set_seed(seeds[2])

  print("[TASK1]: Information about LFSRs:")
  data = []
  for i, name in zip([z1, z2, z3], ["LFSR1", "LFSR2", "LFSR3"]):
    data.append(
      [name, i.get_polynomial(), str(bin(i.get_seed())[2:].ljust(i.get_degree(), '0')) + " " + f"({i.get_seed()})"])
  print(columnar(data, ["Id", "Polynomial", "Initial state"], no_borders=True))

  c = []
  print("[TASK1]: Generating ciphertext from", plaintextfile, "with Geffes generator")
  for y in bitgen(open(plaintextfile, "rb").read()):
    c.append(y ^ gg_combining_function(z1.next_o(), z2.next_o(), z3.next_o()))
  return c


def task3(c):
  """
  TASK3: Break Geffes generator using correlation attack
  :param c: ciphertext in array of bits
  :return: int
  """
  # The polynomials are known to the cryptanalyst
  global z1, z2, z3, Z2S
  q = [0, 0, 0]

  # Check correlation from truth table of the combiner function
  for i in range(8):
    x = list(map(int, bin(i)[2:].zfill(3)))
    f = gg_combining_function(x[0], x[1], x[2])
    for j in range(3):
      if x[j] == f:
        q[j] += 1
  q = [i / 8 for i in q]
  print("[TASK3]: correlation from thruth table z1,z2,z3 -> ", q)

  p0 = 0.6  # We know this value from the probability of the input language (e.g. english ASCII)

  print("[TASK3]: Running correlation attack on z1")
  z1_cand = run_correlation_attack(q[0], p0, c, z1)
  print(f"[TASK3]: z1-candidates: {z1_cand}")
  print("- " * 10)
  print("[TASK3]: Running correlation attack on z3")
  z3_cand = run_correlation_attack(q[2], p0, c, z3)
  print(f"[TASK3]: z3-candidates: {z3_cand}")
  print("- " * 10)
  # Bruteforce z2, now that we know the value of z1 and z3
  print("[TASK3]: Commencing bruteforce of z2")
  print("[TASK3]: Using Shannons entrophy to determine if the plaintext is found")
  for z2_s in range(1, pow(2, z2.get_degree()) - 1):
    for z1_c in z1_cand:
      for z3_c in z3_cand:
        print(f"[TASK3]: Testing {z1_c}, {z2_s} and {z3_c}", end="\r")
        y = []
        z1.set_seed(z1_c)
        z2.set_seed(z2_s)
        z3.set_seed(z3_c)

        for ci in c:
          y.append(ci ^ gg_combining_function(z1.next_o(), z2.next_o(), z3.next_o()))
        # Standard English text usually falls somewhere between 3.5 and 5.0 in shannons entrophy
        ent = entropy(bits2string(y))
        if ent >= 3.5 and ent <= 5:
          print(f"[TASK3]: Possible seeds | Z1:{z1_c} | Z2:{z2_s} | Z3:{z3_c} | entropy of text: {ent:.2f}")

    if DEMO and z2_s == Z2S:  # Speed up the bruteforce, to make the program finish in reasonable time
      print(f"[TASK3]: The correct seed is found, stopping the bruteforce for demonstration purposes...")
      return 0


"""
FUNCTION TO OBTAIN SOME STATISTICS FROM THE ATTACK
"""


def run_statistics(c):
  """
  Get some statistics from the attacks on geffes generator
  :param c: ciphertext in array of bits
  :return:
  """
  # The polynomials are known to the cryptanalyst
  global z1, z2, z3, Z1S, Z3S
  q = [0, 0, 0]

  # Check correlation from truth table of the combiner function
  for i in range(8):
    x = list(map(int, bin(i)[2:].zfill(3)))
    f = gg_combining_function(x[0], x[1], x[2])
    for j in range(3):
      if x[j] == f:
        q[j] += 1
  q = [i / 8 for i in q]

  p0 = 0.6  # We know this value from the probability of the input language (e.g. english ASCII)
  fps_z1, fps_z3, pf = [], [], []

  # Vary the probability of false alarm
  for i in range(1, 50, 3):
    i = i / 1000
    z1_cands = run_correlation_attack(q[0], p0, c[:8000], z1, i)
    z3_cands = run_correlation_attack(q[2], p0, c[:8000], z3, i)
    pf.append(i)
    if Z1S in z1_cands: z1_cands.remove(Z1S)
    if Z3S in z3_cands: z3_cands.remove(Z3S)
    fps_z1.append(len(z1_cands) / (2 ** z1.get_degree() - 1))
    fps_z3.append(len(z3_cands) / (2 ** z3.get_degree() - 1))
  plt.plot(pf, fps_z1, label="R1")
  plt.plot(pf, fps_z3, label="R3")
  plt.xlabel("Pf - value")
  plt.ylabel("False positives")
  plt.legend()
  plt.show()

  fps_z1, fps_z3, l = [], [], []

  # Vary the length of the input file
  for i in range(400, len(c), len(c) // 10):
    z1_cands = run_correlation_attack(q[0], p0, c[:i], z1, 0.01)  # i = 0.01
    z3_cands = run_correlation_attack(q[2], p0, c[:i], z3, 0.01)
    l.append(len(c[:i]))
    if Z1S in z1_cands: z1_cands.remove(Z1S)
    if Z3S in z3_cands: z3_cands.remove(Z3S)
    fps_z1.append(len(z1_cands) / (2 ** z1.get_degree() - 1))
    fps_z3.append(len(z3_cands) / (2 ** z3.get_degree() - 1))
  plt.plot(l, fps_z1, label="R1")
  plt.plot(l, fps_z3, label="R3")
  plt.xlabel("Ciphertext bits")
  plt.ylabel("False positives")
  plt.legend()
  plt.show()
  print(
    f"Mean percent of false positives: L_1:{(sum(fps_z1) / len(fps_z1)):.4f}, L_3:{(sum(fps_z3) / len(fps_z3)):.4f}")


"""
MAIN
"""

if __name__ == "__main__":
  INFILE = os.path.join(os.path.dirname(__file__), INFILE)

  print(" TASK 1 ".center(30, "#"))
  c = task1(INFILE, [Z1S, Z2S, Z3S])  # Geffe's generator

  print(" TASK 3 ".center(30, "#"))
  task3(c)

  if INCLUDE_STATISTICS:  # THIS TAKES QUITE A LOT OF TIME
    print(" RUNNING STATISTICS ".center(30, "#"))
    run_statistics(c)
