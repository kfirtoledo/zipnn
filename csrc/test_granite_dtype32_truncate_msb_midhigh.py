import numpy as np
import split_dtype  # This is your C extension module
import time
from zipnn import ZipNN
import torch

file_path = "granite-3b-code-base.2.bin" 
with open(file_path, 'rb') as file:
    file_bytes = file.read()

zstd = ZipNN(bg_partitions = 1, zstd_threads=1)
                              
#original_bytes = file_bytes[:]
#original_bytes = bytearray(file_bytes[100:108])
original_bytes = bytearray(file_bytes[100000000:1173741824])

slice_length = len(original_bytes[3::4])
zeros = bytearray([0] * slice_length)
original_bytes[3::4] = zeros

slice_length = len(original_bytes[2::4])
zeros = bytearray([0] * slice_length)
original_bytes[2::4] = zeros




original_bytes_saved = bytearray(original_bytes)
print ("len of original bytes ", len(original_bytes)/1024/1024/1024, " GB")

# Function to print the first 8 bytes in bit representation
def print_bit_representation(data, label):
    print(label + ": " + ' '.join(f'{b:08b}' for b in data[:8]))

# Print bit representation of the original data
print_bit_representation(original_bytes, "Original Data")

start_time = time.time()
orig = zstd.compress(original_bytes)
print (f"% of orig {len(orig)/len(original_bytes)}, {time.time() - start_time}")

start_time = time.time()
revert = zstd.decompress(orig)
print (f"% of revert {revert == original_bytes}, {time.time() - start_time}")
zstd = ZipNN(bg_partitions = 1, zstd_threads=1)

# Measure and perform the split
start_time = time.time()
# bit_mode = 1 [bit_ordering],
# byte_mode = 9 [truncate MSB [0,01,,1], 0_00_01_001
# is_review = 0 [no review]
# thread = 1 [one thread]
buf1, buf2, buf3, buf4 = split_dtype.split_dtype32(original_bytes, 0, 9, 0 ,1) # bit_mode = 1 [bit_ordering], 
split_time = time.time() - start_time
start_time = time.time()
comp1 = zstd.compress(buf1)
print (f"% of comp1 {len(comp1)/len(original_bytes)*2}, {time.time() - start_time}")

start_time = time.time()
revert = zstd.decompress(comp1)
print (f"% of revert comp1 {time.time() - start_time}")


if(len(comp1) > (len(original_bytes)/2)):
   len_comp1 = len(original_bytes)/2
else:
   len_comp1 = len(comp1)

print (f"% of comp_all {(len_comp1)/len(original_bytes)}")

# Print bit representations of the splits
print_bit_representation(buf1, "Split Buf1")


# Measure and perform the combine
start_time = time.time()
# bit_mode = 1 [bit_ordering],
# byte_mode = 9 [truncate MSB [0,0,1,1], 0_00_01_001
# thread = 1 [one thread]

buf2 = bytearray()
buf3 = bytearray()
buf4 = bytearray()

combined_bytes = split_dtype.combine_dtype32(buf1, buf2, buf3, buf4, 0, 9, 1)
print_bit_representation(original_bytes_saved, "Original Data")
print_bit_representation(combined_bytes, "Combine")
combine_time = time.time() - start_time
print(f"Split Time: {split_time:.3f} seconds")
print(f"Combine Time: {combine_time:.3f} seconds")

if (original_bytes_saved == combined_bytes):
    print("Test Passed: Original and reverted data are identical.")
else:
    print("Test Failed: Data mismatch.")

