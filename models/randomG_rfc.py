import numpy as np

from legacy.base_rfc import BaseRFC


class RandomGRFC(BaseRFC):
    def create_G(self, **kwargs):
        if self.verbose:
            print("constructing random G")
        return np.array(self.rng.normal(0, 1, (self.N, self.M)))

    # def construct_c(self, patterns, n_washout: int, n_harvest: int):
    #     for pattern in patterns:
    #         for t in range(n_washout):
    #             self.one_step_driving(pattern[t], pattern_id=None)
    #         collected_z = []
    #         for t in range(n_washout, n_washout + n_harvest):
    #             self.one_step_driving(pattern[t], pattern_id=None)
    #             collected_z.append(self.z)
    #         collected_z = np.array(collected_z)
    #
    #         mean_z_squared = np.mean(collected_z ** 2, axis=0)
    #         conceptor_weights = mean_z_squared / (mean_z_squared + self.aperture ** -2)
    #         self.c.append(conceptor_weights)
    #         self.last_training_z_state.append(np.random.normal(size=self.M))
    #     if self.verbose:
    #         print("conceptors constructed", np.shape(self.c))