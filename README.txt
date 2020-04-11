# GeffesGenerator_assessment
Project in the course IMT4124 - Cryptology. This project will use geffes generator (streamcipher) to encrypt a plaintext file. 
Then the program shows an ciphertext only attack-method proposed by T. Siegenthaler in 1985 to recover the plaintext. The program
Is implemented in python3, which is not the fastest programming language out there and certainly not the best choice when it comes
to bruteforcing. On the other hand it is quite easy to debug and write, which is the main reason behind the choice of programming language.

# Installation: 
In a preffered environment, run the follwing command to install the requirements
$ pip install -r requirements.txt
$ python3 task.py # run the program

# TASK 1:
## The task: 

## Implementation:
The function 'task1' reads plaintext from the file specified in the global variable 'INFILE' and returns the ciphertext as an array of bits.
The class 'lfsr' is the implementation of a left shift feedback register (LFSR) and is initialized with an array 'l' that represents the
polynomial connections. It is assumed that '1' is always a part of the polynomail, and thus not a part of the array. Considering the
primitive polynomail 'x⁴+x+1' the corresponding initializing array would be '[4,1]'. A lfsr instance is responsible for its own output bits (period),
genertated by the function 'next_o'. Three LFSRs classes are initialized before the bits of the plaintext file is added to the running key 
sequence generated from the combining function defined in 'gg_combining_function'.

Several different texts are provided in the repository for testing purposes. In order to make the plaintext as realistic to human written text as possible
text from various wikipedia articles have been concatinated into the file 'example.txt'. Smaller files have been derived from this one to test the
program with smaller file sizes. The files '{1,2,3,4,5}000_example.txt' contains the respective first 1000 - 5000 characters.

# TASK 2:
## The task: 


## Implementation:


# TASK 3:
## The task:


## Implementation:

