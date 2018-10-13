from random import choice                                                                                            
from numpy import array, dot, random
import numpy as np
step = lambda x: 0 if x < 0 else 1
data = [ 
    (array([0,0,0,0,0,1]), 0), 
    (array([0,1.414,0,0,1,1]), 1), 
    (array([1.414,0,0,1,0,1]), 1), 
    (array([1.414,1.414,1.414,1,1,1]), 0), 
]

#w = random.rand(6)
w=np.full(6,0.1)
errors = []
eta = 0.2
n = 100
print("The predicted value of every even epoch")
#for i in xrange(n): # this is python 3
for i in range(n): #this is python 2
    x, expected = choice(data)
    result = dot(w, x)
    if i%2 == 0:
        print(dot(w,data[0][0]),dot(w,data[1][0]),dot(w,data[2][0]),dot(w,data[3][0]))

    error = expected - step(result)
    errors.append(error)
    w += eta * error * x

for x, _ in data: 
    result = dot(x, w)
    print("{}: {} -> {}".format(x, result, step(result)))

print(w)
