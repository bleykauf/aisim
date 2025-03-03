from aisim.zern import ZernikeConvention, j_to_nm

noll_mapping_examples = [
    (0, 0),
    (1, 1),
    (1, -1),
    (2, 0),
    (2, -2),
    (2, 2),
    (3, -1),
    (3, 1),
    (3, -3),
    (3, 3),
    (4, 0),
    (4, 2),
    (4, -2),
    (4, 4),
    (4, -4),
    (5, 1),
    (5, -1),
    (5, 3),
    (5, -3),
    (5, 5),
]

ansi_mapping_examples = [
    (0, 0),
    (1, -1),
    (1, 1),
    (2, -2),
    (2, 0),
    (2, 2),
    (3, -3),
    (3, -1),
    (3, 1),
    (3, 3),
    (4, -4),
    (4, -2),
    (4, 0),
    (4, 2),
    (4, 4),
    (5, -5),
    (5, -3),
    (5, -1),
    (5, 1),
    (5, 3),
]


def test_j_to_nm():
    for j, (n, m) in enumerate(noll_mapping_examples):
        j = j + 1
        assert j_to_nm(j, ZernikeConvention.ISO) == (n, m)

    for j, (n, m) in enumerate(ansi_mapping_examples):
        assert j_to_nm(j, ZernikeConvention.ANSI) == (n, m)
