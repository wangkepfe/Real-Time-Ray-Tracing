import numpy as np
import matplotlib.pyplot as plt

def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True

    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq

def halton_sequence2(size, dim, primes):
    seq = []
    for base in primes:
        seq.append([vdc(i, base) for i in range(size)])
    return seq

sampleCount = 60

seq = halton_sequence2(sampleCount, 3, [3,5,7])

arr = np.array(seq)
print(arr)

arr = arr * 5
print(arr)

arr = np.floor(arr)
print(arr)

res = []
for i in range(5):
    current = []
    for j in range(sampleCount):
        if arr[0][j] == i:
            current.append([arr[1][j], arr[2][j]])
    res.append(current)

for i in range(5):
    print(res[i])
    result = np.array(res[i])
    result += 0.1 * i
    plt.scatter(result[:,0], result[:,1])


# result += 0.1
# plt.scatter(result[0,:,0], result[0,:,1])
# result += 0.1
# plt.scatter(result[1,:,0], result[1,:,1])
# result += 0.1
# plt.scatter(result[2,:,0], result[1,:,1])
# result += 0.1
# plt.scatter(result[3,:,0], result[1,:,1])
# result += 0.1
# plt.scatter(result[4,:,0], result[1,:,1])
plt.show()