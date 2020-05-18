import sys
alg_name = sys.argv[1]

filename = "_log_%s.txt" % alg_name
daalLine = "uses Intel® DAAL solver"
sklearnLine = "uses original Scikit-learn solver"
failLine = "uses original Scikit-learn solver, because the task was not solved with Intel® DAAL"

file_log = open(filename, "r")
lines = file_log.readlines()
file_log.close()

countDaalCalls = 0
countSklearnCalls = 0
countDaalFailCalls = 0

for line in lines:
    if daalLine in line:
        countDaalCalls += 1
    if sklearnLine in line:
        countSklearnCalls += 1
    if failLine in line:
        countDaalFailCalls += 1

print("Number of Scikit-learn calls: %d" % countSklearnCalls)
print("Number of daal4py calls: %d" % countDaalCalls)
print("Number of daal4py fail calls:  %d" % countDaalFailCalls)
countAllCalls = countSklearnCalls + countDaalCalls
percentDaalCalls = float(countDaalCalls - countDaalFailCalls) / (countAllCalls) * 100 if countAllCalls else 0
print("Percent of using daal4py: %d %%" % int(percentDaalCalls))
