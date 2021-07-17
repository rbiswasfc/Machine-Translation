import sys
from pathlib import Path


root_dir = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, root_dir)
from utils.data_processor import pad_sents


def test_pad_sents():
    sents = ["it was raining".split(), "the unit test failed due to a bug".split()]
    res = pad_sents(sents, "<pad>")
    assert len(res) == 2
    assert len(res[0]) == 8
    assert res[0][-1] == "<pad>"
    assert res[1][-1] == "bug"
    print("test passed!")


if __name__ == "__main__":
    test_pad_sents()
