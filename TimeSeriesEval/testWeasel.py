
from sktime.classification.dictionary_based import WEASEL
from sktime.datasets import load_gunpoint, load_italy_power_demand


def test_weasel_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)

    # train WEASEL
    weasel = WEASEL(random_state=1, binning_strategy="equi-depth")
    weasel.fit(X_train, y_train)

    score = weasel.score(X_test, y_test)
    print(score)
    assert score >= 0.99
test_weasel_on_gunpoint()