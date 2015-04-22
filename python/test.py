A="1 2 3 4 5 6"
B="0.0505623 -0.089902 -0.0159352 0.101119 0.0973994 -0.0501975"

A=A.split(" ")
B=B.split(" ")
s=0
for j  in range( len(B)):
    a=A[j]
    b=A[j]
    s = s+float(a) * float(b)

print 'result:', s

