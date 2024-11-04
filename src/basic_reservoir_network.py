import numpy as np


class BasicNetwork:
    def __init__(self, N, non_zero=.10, seed=65647437836358831880808032086803839626):
        if not (0 <= non_zero <= 1):
            raise ValueError("percent_non_zero must be between 0 and 100")
        rng = np.random.default_rng(seed)

        self.percent_non_zero = non_zero
        self.N = N

        # Create the matrix W
        total_elements = N * N
        num_non_zero = int(total_elements * non_zero)

        W_flat = np.zeros(total_elements)
        non_zero_indices = rng.choice(total_elements, num_non_zero, replace=False)
        W_flat[non_zero_indices] = rng.uniform(0, 1, num_non_zero)

        self.W = W_flat.reshape(N, N)

        eigenvalues_W = np.linalg.eigvals(self.W)
        spectral_radius = np.max(np.abs(eigenvalues_W))
        print(f"Original Spectral Radius: {spectral_radius}")

        if spectral_radius != 0:  # Avoid division by zero
            self.W = self.W / spectral_radius

        # Create random vectors W_in, r, and b
        self.W_in = rng.random(N)
        self.r = np.array(rng.random(N))
        self.r_initial = self.r.copy()
        self.b = rng.random(N)

    def __repr__(self):
        return (f"RandomMatrixGenerator(N={self.N}, "
                f"percent_non_zero={self.percent_non_zero})")

    def one_step_hallucinating(self):
        print(self.r.shape, self.W.shape)
        self.r = self.W @ self.r

    def one_step_input(self, pattern: float) -> None:
        self.one_step_hallucinating()
        self.r += self.W_in * pattern

    def driving_pattern(self, pattern: list[float], record=False):
        recording = []
        for t in range(len(pattern)):
            self.one_step_input(pattern[t])
            if record:
                recording.append(self.r)
        return recording

    def hallucinate(self, steps, record=False):
        recording = []
        for t in range(steps):
            self.one_step_hallucinating()
            if record:
                recording.append(self.r)
        return recording
