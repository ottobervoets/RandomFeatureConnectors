import numpy as np
from sklearn.decomposition import PCA

from models.base_rfc import BaseRFC

class ReservoirSampleRFC(BaseRFC):

    def create_F(self, **kwargs):
        if self.verbose:
            print("Sampeling from activations")
        patterns = kwargs.get("training_patterns")
        washout = kwargs.get("washout")
        n_adapt = kwargs.get("n_adapt")
        sample_rate = kwargs.get("sample_rate")
        F = []
        for name, pattern in patterns.items():
            r = np.random.normal(0, 0.5, size=self.N)
            for t in range(washout):
                r = np.tanh(self.W @ r + self.W_in @ np.atleast_1d(pattern[t]) + self.b)
            t = washout
            while(True):
                t += 1
                r = np.tanh(self.W @ r + self.W_in @ np.atleast_1d(pattern[t]) + self.b)
                if t%sample_rate == 0:
                    F.append(r)
                    if len(F)%(int(self.M/len(patterns))) == 0:
                        print(f"Map has length {len(F)} after {t} timesteps, continue to pattern {name}")
                        break

        if self.verbose:
            print(f"{len(F)} components are taken from the activations, the remaining {self.M-len(F)} will be sampled randomly")
        F.extend(ReservoirSampleRFC.sample_unit_vectors(self.M-len(F), self.N).tolist())

        # F = np.random.normal(0,0.5, (self.M,self.N))

        F = self.scale_to_unit_norm(F)
        self.print_mean_dot(F)
        F = np.array(F)
        return F.T

    def scale_to_unit_norm(self, vectors):
        unit_vectors = []
        for vector in vectors:
            norm = np.linalg.norm(vector)  # Calculate the norm (magnitude) of the vector
            if norm == 0:
                unit_vectors.append(vector)  # Keep zero vectors unchanged
            else:
                unit_vectors.append(np.array(vector) / norm)  # Scale the vector
        return unit_vectors

    @staticmethod
    def sample_unit_vectors(n_rand, dimension):
        # Step 1: Sample random vectors from a normal distribution
        random_vectors = np.random.normal(size=(n_rand, dimension))


        # Step 2: Compute the norms of the vectors
        norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)

        # Step 3: Normalize the vectors to have unit norm
        unit_vectors = random_vectors / norms
        # print(np.shape(unit_vectors), "unti vectors")

        return unit_vectors

    def print_mean_dot(self, F):
        total, counter, pos1, pos2 = 0,0,0,0

        for pos1 in range(len(F)):
            for pos2 in range(pos1,len(F)):
                counter += 1
                total += np.dot(F[pos1], F[pos2])
        print(f"total dot product {total/counter} over {counter} dot products")


