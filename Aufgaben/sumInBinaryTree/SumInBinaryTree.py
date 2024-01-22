import math
import sys

try:
    inputt = int(input("Gib eine ganze Zahl ein: "))
    print("Du hast die ganze Zahl", inputt, "eingegeben.")
except ValueError:
    print("Fehler: Das ist keine ganze Zahl.")
    sys.exit()
    
sum = 0
for i in range(0,math.floor(math.log2(inputt)),1):
    sum +=  math.floor(inputt/2**i)
    print(f"node: {i} value  = {math.floor(inputt/2**i)}")

sum =sum +1
print(f"sum = {sum}")









