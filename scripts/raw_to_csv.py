# Raw data from web to CSV file.
# Usage: python3 raw_to_csv.py LLCP2015.txt cols-2015.csv > LLCP2015.csv

import sys
import csv


class Column:
    def __init__(self, start, name, length):
        self.start = int(start)
        self.name = name
        self.length = int(length)


def line_to_row(line, cols):
    """
    split a line into a list
    """
    row = []
    for col in cols:
        val = line[(col.start-1):(col.start-1+col.length)]
        val = val.strip()
        row.append(val)
    return row


def main(raw_file_name, cols_file_name):
    rows = []

    print("Reading column spec file...", file=sys.stderr)
    with open(cols_file_name, errors='replace') as col_file:
        # read csv into a list of column specifications
        cols = []
        reader = csv.reader(col_file)
        for row in reader:
            col = Column(row[0], row[1], row[2])
            cols.append(col)

        print("Reading raw datafile...", file=sys.stderr)
        with open(raw_file_name, errors='replace') as raw_file:
            i = 0
            for line in raw_file:
                row = line_to_row(line, cols)
                rows.append(row)
                i += 1
                print("Reading line", i, end='\r', file=sys.stderr)

    print("Writing CSV file...", file=sys.stderr)
    writer = csv.writer(sys.stdout)
    writer.writerow([col.name for col in cols])
    writer.writerows(rows)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

