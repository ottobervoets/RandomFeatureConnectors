# Meeting notes

### 15 october
- $F'G$ does not work with back transpose. I dont know why at this point. but it is un able to converge to nice conceptors. TODO: work this out
 
- How to compare chaotic systems
- Do we care about G and G = G_tilde @ W, the filldness of G/W?

Figure 1 (Construction of F is random)

    n_harvest = 400
    washout = 500
    learning_rate_c = 0.5
    beta_W_out = 0.01
    beta_G = 1
    beta_D = 0.001
    aperture = 8
    spectral_radius = 1.4
    N = 500
    M = 1000

    # self.W = np.array(self.rng.normal(0, 1, (N, N)))

    # Create random vectors W_in, r, and b
    self.W_in = np.array(self.rng.normal(0, 1.2, (self.N, self.signal_dim)))
    self.W_out = np.array(self.rng.random((self.signal_dim, self.N)))

    self.b = self.rng.normal(0, 0.2, self.N)
    # self.D = self.rng.normal(0, 1, (self.N, self.M))  # random initialize D
    self.D = self.rng.normal(0, 1, self.M)  # random initialize D

Figure 2
4 times higher NRMSE
