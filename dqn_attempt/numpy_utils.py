import numpy as np

def shorten_array(array, chunk_size):
    # Ensure the array can be divided into chunks of the specified size
    if len(array) % chunk_size != 0:
        raise ValueError("Array length is not evenly divisible by chunk size")
    
    # Split the array into chunks
    chunks = np.split(array, len(array) // chunk_size)
    
    # Sum each chunk
    chunk_sums = [np.sum(chunk) for chunk in chunks]
    
    return np.array(chunk_sums)

# def split_array(array):
#     # Ensure the array can be divided into chunks of the specified size
#     if len(array) % 10 != 0:
#         raise ValueError("Array length is not evenly divisible by number of houses")
    
#     # Split the array into chunks
#     chunks = np.split(array, len(array) // 10)
    
#     # Sum each chunk
    
#     return chunks