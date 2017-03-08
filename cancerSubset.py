# cancerSubset.py
# usage: python3 stateCancer.py llcp_filename > output_filename
import sys
inputFileName = sys.argv[1]
cancerCol = int(sys.argv[2])
with open(inputFileName, errors='replace') as f:
    for line in f:
        if u'\uFFFD' not in line:
            print(
                line[cancerCol-1],
                line[89],
                line[100],
                line[104],
                line[105],
                line[106],
                line[107],
                line[108],
                line[112],
                line[113],
                line[114],
                line[115],
                line[116],
                sep = ','
            )
