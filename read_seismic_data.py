import numpy as np

def get_random_chunk(data, chunk_size=32):
    w = np.random.randint(0, data.shape[0] - 5)
    x = np.random.randint(0, data.shape[1] - chunk_size)
    y = np.random.randint(0, data.shape[2] - chunk_size)
    z = np.random.randint(200, data.shape[3])

    chunk = np.array(data[w, x:x + chunk_size, y:y + chunk_size, z])

    return chunk

def get_random_multilayer_chunk(data, chunk_size=32, layers=5):
    w = np.random.randint(0, data.shape[0] - 5)
    x = np.random.randint(0, data.shape[1] - chunk_size)
    y = np.random.randint(0, data.shape[2] - chunk_size)
    z = np.random.randint(0, data.shape[3] - layers)

    chunk = np.array(data[w, x:x + chunk_size, y:y + chunk_size, z:z + layers])
    chunk = np.transpose(chunk, (2, 0, 1))

    return chunk

def random_rotate_and_mirror(chunk):
    k = np.random.randint(0, 4)
    chunk = np.rot90(chunk, k).copy()

    if np.random.randint(0, 2):
        chunk = chunk[::-1, :]

    return chunk

def random_rotate_and_mirror_multilayer(chunk):
    k = np.random.randint(0, 4)
    chunk = np.rot90(chunk, k, axes=(1,2)).copy()

    if np.random.randint(0, 2):
        chunk = chunk[:, ::-1, :]

    return chunk

with open("SegActi-45x201x201x614.bin", "rb") as f:
    # Read dimensions (3 integers as big-endian)
    w = 45
    x = 201
    y = 201
    z = 614
    num_elements = w * x * y * z
    data = np.frombuffer(f.read(num_elements * 4), dtype="f4").reshape(w, x, y, z)

data = np.array(data)
data /= np.abs(data).max((2, 3), keepdims=True) / 2

def get_chunk(chunk_size=32):
    return random_rotate_and_mirror(get_random_chunk(data, chunk_size))

def get_chunks(n_chunks, chunk_size=32):
    return np.stack([np.expand_dims(get_chunk(chunk_size), 0) for _ in range(n_chunks)])

def get_multilayer_chunk(chunk_size=32, layers=5):
    return random_rotate_and_mirror_multilayer(get_random_multilayer_chunk(data, chunk_size, layers))

def get_multilayer_chunks(n_chunks, chunk_size=32, layers=5):
    return np.stack([get_multilayer_chunk(chunk_size, layers) for _ in range(n_chunks)])
