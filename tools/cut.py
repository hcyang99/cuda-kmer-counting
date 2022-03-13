import sys

f = open("data/reference.fa", "r")
for line_cnt in range(80000):
    line = f.readline()
    sys.stdout.write(line)
f.close()