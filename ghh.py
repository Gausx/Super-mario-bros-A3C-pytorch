import numpy as np

a = 1
b = 1

if a == 1:
    print("a")
elif b == 1:
    print("b")


def rules(tiles, action):
    COMPLEX_MOVEMENT = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
        ['left', 'A'],
        ['left', 'B'],
        ['left', 'A', 'B'],
        ['down'],
        ['up'],
    ]

    if (action == 2 or action == 4) and 170 in tiles and np.argwhere(tiles == 170)[0, 0] == 12 and tiles[
        13, np.argwhere(tiles == 170)[0, 1]] == 0:
        return True
    else:
        return False



