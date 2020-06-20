import os
from functools import reduce

directory_path = os.path.dirname(os.path.realpath(__file__))

with open(f"{directory_path}/in.txt", "r") as file_in:
    raw_data_in = list(map(lambda line: list(map(int, line.split())), file_in.read().splitlines()))

with open(f"{directory_path}/out.txt", "r") as file_out:
    out_lines = file_out.read().splitlines()
    out_lines_concat = reduce(lambda accumulated, current: accumulated + current, out_lines)
    out_lines_concat_casted = map(int, out_lines_concat)

    raw_data_out = list(out_lines_concat_casted)

with open(f"{directory_path}/test.txt", "r") as file_test:
    raw_data_test = list(map(lambda line: list(map(int, line.split())), file_test.read().splitlines()))

assert raw_data_in and raw_data_out, "No IN data or no OUT data"
assert len(raw_data_in) == len(
    raw_data_out), f"The amount of IN data ({len(raw_data_in)}) does not much the number of OUT data ({len(raw_data_out)})"
