#!/usr/bin/env python3 
import math

from scipy.stats import norm  # norm.cdf(1.96) and norm.ppf(norm.cdf(1.96))


# initialized with the polynomial, and the seed are added dynamically
class lfsr(list):
  def get_seed(self):
    return self.start_val

  def set_seed(self, seed):
    self.reg = self.start_val = seed

  def next_o(self):
    out = self.reg & 1  # get the LS bit
    b = ((self.reg >> self[0] - 1) & 1)  # get the first bit to XOR
    for p in self[1:]:  # loop the polynomial degrees (index of bits to xor)
      b ^= ((self.reg >> p - 1) & 1)  # shift the register accordingly and get the LSbit to XOR with the other ones
    self.reg = (self.reg >> 1 | b << (max(self) - 1))  # Shift the register and apply the new bit (will also pad)
    # TODO: How to determine the length of the LFSR register?
    # Assumes it is as long as the degree of the polynomial-1
    return out

  def get_degree(self):
    return max(self)

  def get_x_rounds(self, x=1):
    o_a = []
    for i in range(x):
      o_a.append(self.next_o())
    return o_a


def gg_combining_function(z1, z2, z3):
  return (z3 ^ (z1 & z2) ^ (z2 & z3))


def run_correlation_attack(qi, p0, c, z):
  candidates = []
  # TODO: do we need all of theese calculations
  pe = 1 - (p0 + qi) + 2 * p0 * qi
  l = len(c)  # This can be varied, depending on how much you are reading, and will influence the rest
  pf = 0.05  # This can be adjusted, or the Pm can be determined, and the whole thing reversed a bit (might be better?)
  T = norm.ppf(1 - pf) * math.sqrt(l)  # and norm.ppf(norm.cdf(1.96))
  pm = 1 - norm.cdf((l * (2 * pe - 1) - T) / (math.sqrt(4 * l * pe * (1 - pe))))
  Ri = pow(2, z.get_degree())
  print("pe: {}\npm: {}\npf: {}\nl: {}\nT: {}\nRi: {}".format(pe, pm, pf, l, T, Ri))

  for i in range(1, Ri):
    z.set_seed(i)
    ham_d = 0
    for n in range(l):
      ham_d += c[n] ^ z.next_o()

    if l - (2 * ham_d) >= T:
      candidates.append(i)

  return candidates


# Generate bits from an input (bytes)
def bitgen(x):
  for c in x:
    for i in range(8):
      yield int((c & (0x80 >> i)) != 0)


def task1():
  print("[TASK1]: Generating keystream")
  z1 = lfsr([9, 6, 4, 3, 1])  # x^11+x^7+x^3+x+1
  z2 = lfsr([10, 7, 3, 1])  # x^11+x^7+x^3+x+1
  z3 = lfsr([8, 6, 5, 2, 1])  # x^11+x^7+x^3+x+1

  # z2 = lfsr([20, 15, 12, 8, 6, 5])                      # x^20+x^15+x^12+x^8+x^6+x^5+1
  # z3 = lfsr([15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1])  # x^15+x^14+x^13+x^12+x^11+x^9+x^8+x^7+x^5+x^4+x^2+x+1
  # set the key / seed values for the LFSR's (needs to be less than 2^(length of LFSR)
  # THis is the key (such secrecy):
  z1.set_seed(1)
  z2.set_seed(2)
  z3.set_seed(3)

  c = []

  # This is fast ish?
  for y in bitgen(open("500_example.txt", "rb").read()):  # This is a generator object
    c.append(y ^ gg_combining_function(z1.next_o(), z2.next_o(), z3.next_o()))
  return c


def task2(c):
  z1 = lfsr([9, 6, 4, 3, 1])  # x^11+x^7+x^3+x+1
  z2 = lfsr([10, 7, 3, 1])  # x^11+x^7+x^3+x+1
  z3 = lfsr([8, 6, 5, 2, 1])  # x^11+x^7+x^3+x+1

  q = [0, 0, 0]

  # Check correlation from truth table of the boolean combiner function
  for i in range(8):
    x = list(map(int, bin(i)[2:].zfill(3)))
    f = gg_combining_function(x[0], x[1], x[2])
    for j in range(3):
      if x[j] == f:
        q[j] += 1
  q = [i / 8 for i in q]
  print(q)
  p0 = 0.6  # TODO We know this value from the probability of the input language (feks. english ASCII)

  # Can we multithread this --> How ti get the return value?
  z1_cand = run_correlation_attack(q[0], p0, c, z1)  # Do we need just one?
  z3_cand = run_correlation_attack(q[2], p0, c, z3)

  # Bruteforce z2, now that we know the value of z1 and z3
  print("[TASK2]: Commencing bruteforce of z2")
  print(z1_cand, z3_cand)
  for z1_s in range(1, pow(2, z2.get_degree()) + 1):
    for z1_c in z1_cand:
      for z3_c in z3_cand:
        crnt_key_strm = []
        z1.set_seed(z1_c)
        z2.set_seed(z1_s)
        z3.set_seed(z3_c)

        for _ in range(len(c)):
          crnt_key_strm.append(gg_combining_function(z1.next_o(), z2.next_o(), z3.next_o()))
        if crnt_key_strm == c:
          print("[TASK2]: The seeds are:\nz1:{}\nz2:{}\nz3:{}".format(z1.get_seed(), i, z3.get_seed()))
          return
    print(z1_s)


# The program starts here
if __name__ == "__main__":
  c = task1()
  print(sum([i ^ 0 for i in c]) / len(c))  # This cannot be 0.5
  task2(c)
