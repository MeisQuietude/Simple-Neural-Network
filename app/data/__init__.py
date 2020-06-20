from numpy import array

from ._parse import raw_data_in, raw_data_out, raw_data_test

# Append 1 as additional parameter to avoid extra multiplication by 0
raw_data_in = [set_ + [1] for set_ in raw_data_in]
raw_data_test = [set_ + [1] for set_ in raw_data_test]

# Create ndarray of each data set for export it
data_in = array(raw_data_in)
data_out = array([raw_data_out]).T
data_test = array(raw_data_test)

print(f"Train cases: {len(data_in)}")
if len(data_test):
    print(f"Test cases:  {len(data_test)}")
print()
