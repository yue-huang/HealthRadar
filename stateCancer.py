# stateCancer.py
# usage: python3 stateCancer.py llcp_filename > output_filename
import sys
inputFileName = sys.argv[1]
cancerCol = int(sys.argv[2])
stateCancer = list()
with open(inputFileName, errors='replace') as f:
    for line in f:
        if u'\uFFFD' not in line:
            print('{0},{1}'.format(line[0:2], line[cancerCol-1]))
