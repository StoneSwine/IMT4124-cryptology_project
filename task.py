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
  pf = 0.01  # This can be adjusted, or the Pm can be determined, and the whole thing reversed a bit (might be better?)
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
  print("[TASK1]: Generating ciphertext")
  z1 = lfsr([10, 7, 3, 1])
  z2 = lfsr([20, 15, 12, 8, 6, 5])
  z3 = lfsr([11, 7, 3, 1])

  # set the key / seed values for the LFSR's (needs to be less than 2^(length of LFSR)
  # THis is the key (such secrecy):
  z1.set_seed(602)
  z2.set_seed(148)
  z3.set_seed(901)

  c = []

  # This is fast ish?
  for y in bitgen(open("example.txt", "rb").read()):  # This is a generator object
    c.append(y ^ gg_combining_function(z1.next_o(), z2.next_o(), z3.next_o()))
  return c


def bits2string(bits=None):  # This function is taken from https://stackoverflow.com/a/10238140
  chars = []
  for b in range(len(bits) // 8):
    byte = bits[b * 8:(b + 1) * 8]
    chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
  return ''.join(chars)


def entropy(string):  # This function is taken from https://stackoverflow.com/a/2979208
  # get probability of chars in string
  prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
  # calculate the entropy
  entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
  return entropy


def task2(c):
  z1 = lfsr([10, 7, 3, 1])
  z2 = lfsr([20, 15, 12, 8, 6, 5])                      # x^20+x^15+x^12+x^8+x^6+x^5+1
  z3 = lfsr([11, 7, 3, 1])
  q = [0, 0, 0]

  # Check correlation from truth table of the boolean combiner function
  for i in range(8):
    x = list(map(int, bin(i)[2:].zfill(3)))
    f = gg_combining_function(x[0], x[1], x[2])
    for j in range(3):
      if x[j] == f:
        q[j] += 1
  q = [i / 8 for i in q]

  p0 = 0.6  # TODO We know this value from the probability of the input language (e.g. english ASCII)

  # Can we multithread this --> How to get the return value -> Is there something easier?
  print("[TASK2]: Running correlation attack on z1")
  z1_cand = run_correlation_attack(q[0], p0, c, z1)

  print("[TASK2]: Running correlation attack on z3")
  z3_cand = run_correlation_attack(q[2], p0, c, z3)

  print(z1_cand)
  print(z3_cand)

  # Bruteforce z2, now that we know the value of z1 and z3
  print("[TASK2]: Commencing bruteforce of z2")
  for z2_s in range(1, pow(2, z2.get_degree()) + 1):
    for z1_c in z1_cand:
      for z3_c in z3_cand:
        y = []
        z1.set_seed(z1_c)
        z2.set_seed(z2_s)
        z3.set_seed(z3_c)

        for ci in c:
          y.append(ci ^ gg_combining_function(z1.next_o(), z2.next_o(), z3.next_o()))
        # Standard English text usually falls somewhere between 3.5 and 5.0 in shannons entrophy
        if 5.0 >= entropy(bits2string(y)) >= 3.5:
          print("[TASK 2]: Candidate seeds - Z1:{} Z2:{} Z3:{} ".format(z1_c, z2_s, z3_c))
    print(z2_s)

# The program starts here
if __name__ == "__main__":
  c = task1()
  task2(c)
