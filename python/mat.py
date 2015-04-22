import numpy

A="1,2,3,4,5,6"
B="1,2,3,4,5,5"

a=[]
for item in A.split(","):
    a.append(float(item))

b=[]
for item in B.split(","):
    b.append(float(item))

a=numpy.array(a).reshape(3,2)
b=numpy.array(b).reshape(3,2).transpose()
c=numpy.dot(a,b)

print(a)
print
print(b)
print
print(c)



