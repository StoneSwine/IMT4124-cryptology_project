

"""
[TASK 4]:
  This code help to find an improved combiner function (less corrleation), than the one in Geffe's generator
"""

BALANCED = True
BALANCED_ERRORMARGIN = 0

x_vals = []
for i in range(8):
  x_vals.append(list(map(int, bin(i)[2:].zfill(3))))

for f_int in range(int("1"*8,2)):
  bin_f = list(map(int, bin(f_int)[2:].zfill(8)))
  if BALANCED and (sum(bin_f) != 4+BALANCED_ERRORMARGIN or sum(bin_f) != 4-BALANCED_ERRORMARGIN): # Balancedness criteria
    continue
  else:
    q = [0, 0, 0]
    for x, f in zip(x_vals, bin_f):
      for i in range(3):
        if x[i] == f:
          q[i] += 1

    q = [i / 8 for i in q]
    if q.count(0.5) == 2 and q.count(1) == 0 and q.count(0.0) == 0: # Filter out bad correlations
      print("Correlation from thruth table z1,z2,z3 -> ", q)
      print(bin_f,f_int)


