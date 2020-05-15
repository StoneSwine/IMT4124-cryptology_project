class lfsr(list):
  """initialized with the polynomial, and the seed are added dynamically
  """

  def __init__(self, l):
    super().__init__(l)
    self.degree = max(l)

  def get_polynomial(self):
    str = "1"
    for i in reversed(self):
      str += "+x^{}".format(i)
    return str

  def get_seed(self):
    return self.start_val

  def set_seed(self, seed):
    self.reg = self.start_val = seed

  def next_o(self):
    out = self.reg & 1  # get the LS bit
    b = ((self.reg >> self[0] - 1) & 1)  # get the first bit to XOR
    for p in self[1:]:  # loop the polynomial degrees (index of bits to xor)
      b ^= ((self.reg >> p - 1) & 1)  # shift the register accordingly and get the LSbit to XOR with the other ones
    self.reg = (self.reg >> 1 | b << (self.degree - 1))  # Shift the register and apply the new bit (will also pad)
    # TODO: How to determine the length of the LFSR register?
    # Assumes it is as long as the degree of the polynomial-1
    return out

  def get_degree(self):
    return self.degree

  def get_x_rounds(self, x=1):
    o_a = []
    for _ in range(x):
      o_a.append(self.next_o())
    return o_a
