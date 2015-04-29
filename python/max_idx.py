import sys

max_idx=0
for line in sys.stdin:
    array=line.split(' ')
    for item in array:
        if ":" in item:
            idx, _ = item.split(":",1)
            idx=int(idx)
            if idx>max_idx:
                max_idx = idx

print 'max_idx:',max_idx
