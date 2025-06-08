import numpy as np

# avoid array index error & overflows
# now we have dx directly
class ReputationGrid:
    def __init__(self, R=1.0, n=30):
        self.n = n
        self.R = R
        self.dx = R / n
        self.r = np.linspace(0, R, n + 1)

    def zeros(self, *extra_dims):
        return np.zeros((self.n + 1, *extra_dims))

    def ones(self, *extra_dims):
        return np.ones((self.n + 1, *extra_dims))

    def full(self, value, *extra_dims):
        return np.full((self.n + 1, *extra_dims), value)

    def shape(self, *extra_dims):
        return (self.n + 1, *extra_dims)

    def linspace(self):
        return self.r

    def size(self):
        return self.n + 1

    def all_states(self):
        return range(self.n + 1)

    def next_state(self, i):
        return min(i + 1, self.n)

    def prev_state(self, i):
        return max(i - 1, 0)

    def clip_index(self, i):
        return min(max(i, 0), self.n)

    def rep_index(self, value):
        idx = np.argmin(np.abs(self.r - value))
        return idx