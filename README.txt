# GeffesGenerator_assessment
Project in the course IMT4124 - Cryptology. This project will use geffes generator (streamcipher) to encrypt a plaintext file. Then the program shows an ciphertext only attack-method proposed by T. Siegenthaler in 1985 to recover the plaintext. The programIs implemented in python3, which is not the fastest programming language out there and certainly not the best choice when it comesto bruteforcing. On the other hand it is quite easy to debug and write, which is the main reason behind the choice of programming language.

# Installation:
In a preffered environment, run the follwing command to install the requirements
$ pip install -r requirements.txt
$ python3 task.py #run the program

# TASK 1:
## The task:
## Implementation:
The function 'task1' reads plaintext from the file specified in the global variable 'INFILE' and returns the ciphertext as an array of bits.Where the key (initial state) specified in the global variables 'Z{1,2,3}S'.The class 'lfsr' is the implementation of a left shift feedback register (LFSR) and is initialized with an array 'l' that represents thepolynomial connections. It is assumed that '1' is always a part of the polynomail, and thus not a part of the array. Considering theprimitive polynomail 'x⁴+x+1' the corresponding initializing array would be '[4,1]'. A lfsr instance is responsible for its own output bits (period),genertated by the function 'next_o'. Three LFSRs classes are initialized before the bits of the plaintext file is added to the running key sequence generated from the combining function defined in 'gg_combining_function'.Several different texts are provided in the repository for testing purposes. In order to make the plaintext as realistic to human written text as possibletext from various wikipedia articles have been concatinated into the files '{1,2,3,4,5}000_example.txt'. The files contains 1000 - 5000 characters, to demonstrate different lengths of input files.

# TASK 3:
## The task:
## Implementation

def run_correlation_attack(qi, p0, l, pf, ri):
...   pe = 1 - (p0 + qi) + 2 * p0 * qi
...   T = norm.ppf(1 - pf) * math.sqrt(l)
...   pm = 1 - norm.cdf((l * (2 * pe - 1) - T) / (math.sqrt(4 * l * pe * (1 - pe))))
...   n = (((1/math.sqrt(2))*math.sqrt(math.log(2**(ri-1))) + pm*math.sqrt(pe*(1-pe))) / (pe - 0.5))**2
...   print(n)

run_correlation_attack(0.75, 0.6, 8000, 0.002, 20)
0.75 0.6
[TASK3]: Information about variables:
	pe: 0.5499999999999998
	pm: 5.415035086997477e-10
	pf: 0.002
	l: 8000 bit
	T: 257.4

2633.959286680847
run_correlation_attack(0.75, 0.6, 8000, 0.002, 10)
0.75 0.6
[TASK3]: Information about variables:
	pe: 0.5499999999999998
	pm: 5.415035086997477e-10
	pf: 0.002
	l: 8000 bit
	T: 257.4

1247.6649253885355
run_correlation_attack(0.75, 0.6, 8000, 0.002, 11)
0.75 0.6
[TASK3]: Information about variables:
	pe: 0.5499999999999998
	pm: 5.415035086997477e-10
	pf: 0.002
	l: 8000 bit
	T: 257.4

1386.2943615211143

# TASK 4:
## The task:
## Implementation

