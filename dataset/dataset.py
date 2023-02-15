class Dataset:
    def __init__(self, x, y=None, prop_test=None, k_folds=None) -> None:
        self.x = x
        self.y = y

    def compute_gram_matrix(self, kernel):
        pass
