import numpy as np
from sklearn.decomposition import PCA

# from models.base_rfc import BaseRFC
from models.base_rfc import BaseRFC

class PCARFC(BaseRFC):

    def create_F(self, **kwargs):
        patterns = kwargs.get("training_patterns")
        washout = kwargs.get("washout")
        n_adapt = kwargs.get("n_adapt")
        max_n_components = kwargs.get("max_n_features")
        if len(patterns) * max_n_components > self.M:
            UserWarning("M is to low to accomodate the desired number of components")
            max_n_components = int(np.floor(self.M/len(patterns)))
        F = []
        for name, pattern in patterns.items():
            R = [] #matrix containing reservoir states
            r = np.random.normal(0, 0.5, size=self.N)
            for t in range(washout):
                r = np.tanh(self.W @ r + self.W_in @ np.atleast_1d(pattern[t]) + self.b)

            for t in range(washout, washout+n_adapt):
                r = np.tanh(self.W @ r + self.W_in @ np.atleast_1d(pattern[t]) + self.b)
                R.append(r)

            R = np.array(R)
            corr = R.T @ R
            pca = PCA()
            pca.fit(corr)
            F.extend(pca.components_[0:max_n_components])
        if self.verbose:
            print(f"{len(F)} components are taken from the pca, the remaining {self.M-len(F)} will be sampled randomly")

        F.extend(PCARFC.sample_unit_vectors(self.M-len(F), self.N).tolist())
        F = np.array(F)
        print(F.shape)
        return F.T
    @staticmethod
    def sample_unit_vectors(n_rand, dimension):
        # Step 1: Sample random vectors from a normal distribution
        random_vectors = np.random.normal(size=(n_rand, dimension))


        # Step 2: Compute the norms of the vectors
        norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)

        # Step 3: Normalize the vectors to have unit norm
        unit_vectors = random_vectors / norms
        print(np.shape(unit_vectors), "unti vectors")

        return unit_vectors


