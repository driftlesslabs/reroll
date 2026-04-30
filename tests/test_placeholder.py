import numpy as np


def test_write_memory():

    from reroll._fast_random import _BIT_GENERATOR, _state_bytes

    print("\n\n$$$ _state_bytes $$$")
    print(_state_bytes)

    print("\n\n$$$ _BIT_GENERATOR $$$")
    print(_BIT_GENERATOR.state)

    state_array = np.empty(shape=[4], dtype=np.uint64)
    bstate = _BIT_GENERATOR.state["state"]

    val_128 = bstate["state"]
    state_array[0] = val_128 & 0xFFFFFFFFFFFFFFFF
    state_array[1] = val_128 >> 64
    val_128 = bstate["inc"]
    state_array[2] = val_128 & 0xFFFFFFFFFFFFFFFF
    state_array[3] = val_128 >> 64

    print("\n\n$$$ state_array $$$")
    print(state_array)

    print("\n\n$$$ _state_bytes.view $$$")
    print(_state_bytes.view(np.uint64))


if __name__ == "__main__":
    test_write_memory()
