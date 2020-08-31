from acrobotics.urdfio import import_urdf, export_urdf

TEST_DIR = "./tests"  # is this robust??


def test_complete_in_out():
    scene = import_urdf("example", TEST_DIR)
    export_urdf(scene, "example_after", TEST_DIR)

    file_before = []
    with open(TEST_DIR + "/example.urdf") as f1:
        file_before = f1.readline()

    file_after = []
    with open(TEST_DIR + "/example_after.urdf") as f2:
        file_after = f2.readline()

    assert len(file_after) == len(file_after)

    for lb, la in zip(file_before, file_after):
        assert lb == la

