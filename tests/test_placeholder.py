import numpy as np


def test_write_memory():

    from reroll._fast_random import FastGenerator

    FG = FastGenerator()

    print("\n\n$$$ _state_bytes $$$")
    print(FG._state_bytes)

    print("\n\n$$$ _BIT_GENERATOR $$$")
    print(FG._bit_generator.state)

    state_array = np.empty(shape=[4], dtype=np.uint64)
    bstate = FG._bit_generator.state["state"]

    val_128 = bstate["state"]
    state_array[0] = val_128 & 0xFFFFFFFFFFFFFFFF
    state_array[1] = val_128 >> 64
    val_128 = bstate["inc"]
    state_array[2] = val_128 & 0xFFFFFFFFFFFFFFFF
    state_array[3] = val_128 >> 64

    print("\n\n$$$ state_array $$$")
    print(state_array)

    print("\n\n$$$ _state_bytes.view $$$")
    print(FG._state_bytes.view(np.uint64))

    # There are 4 values in state_array, which correspond to 4 values
    # within _state_bytes.view(np.uint64), in a contiguous block but
    # not necessarily in the same order.  Find them
    viewer = FG._state_bytes.view(np.uint64)
    positions = []
    for j in range(4):
        target = state_array[j]
        for k in range(16):
            if viewer[k] == target:
                break
        if k >= 15:
            raise ValueError("target failed")
        positions.append(k)
    print("\n\n$$$ positions $$$")
    print(positions)

    selection_slice = slice(min(positions), max(positions) + 1)
    print("\n\n$$$ slice $$$")
    print(selection_slice)


if __name__ == "__main__":
    test_write_memory()
