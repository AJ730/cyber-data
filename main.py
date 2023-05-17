import numpy as np


class SMOTE:

    def __init__(self, samples, N, T):
        self.num_attrs = len(samples[0])  # Number of Attributes
        self.new_index = 0  # Number of Attributes
        self.synthetic = []
        self.samples = samples

        if N < 100:
            T = (N / 100) * T
            N = 100

        self.N = int(N / 100)
        self.T = T

    def populate(self, synthetic, N, i, nnarray, k, num_attrs):
        while N != 0:
            nn = np.random.randint(1, k + 1)
            for attr in range(0
                    , num_attrs + 1):
                dif = self.samples[nnarray[nn][attr]] - self.samples[i][attr]
                gap = np.random.random()
                synthetic[self.new_index][attr] = self.samples[i][attr] + gap * dif

            self.new_index += 1
            N = N - 1
        return synthetic

    def get_nearest_neighbors(self, samples, i, k):
        distances = []
        for j in range(len(samples)):
            if j != i:
                dist = self.euclidean_distance(samples[i], samples[j])
                distances.append((j, dist))

        distances.sort(key=lambda x: x[1])
        neighbors = [distances[m][0] for m in range(k)]

        return neighbors

    @staticmethod
    def euclidean_distance(v1, v2):
        return sum((p - q) ^ 2 for p, q in zip(v1, v2)) ** 1 / 2


if __name__ == '__main__':
    ### Smote###

    import random

    # Input: Number of Minority samples T; amount of SMOTE N%; Number of Nearest neigbours k; samples: Array for original minority class samples
    # class SMOTE(T, N, k, samples):
    #     if N < 100:
    #         T = (N/100) * T
    #         N = 100
    #
    #     N = int(N/100)
    #
    #
    #     for i in range(1, T+1):
    #         nn_array = get_nearest_neighbors(samples, i, k)
    #         populate(samples, synthetic, N, i, nn_array, k, num_attrs, new_index)
    #
    #
    #     def populate(samples, synthetic, N, i, nnarray, k, num_attrs):
    #         while N!=0:
    #             nn = np.random.randint(1, k+1)
    #             for attr in range(1, num_attrs+1):
    #                 dif = samples[nnarray[nn][attr]] - samples[i][attr]
    #                 gap = np.random.random()
    #                 synthetic[newindex][attr] = samples[i][attr] + gap*dif
    #
    #             newindex+=1
    #             N = N-1
    #         return synthetic
    #
    #
    #     def get_nearest_neighbors(samples, i, k):
    #         distances = []
    #         for j in range(len(samples)):
    #             if j != i:
    #                 dist = euclidean_distance(samples[i], samples[j])
    #                 distances.append((j, dist))
    #
    #         distances.sort(key=lambda x: x[1])
    #         neighbors = [distances[m][0] for m in range(k)]
    #
    #         return neighbors
    #
    #     def euclidean_distance(v1, v2):
    #         return sum((p-q)^2 for p, q in zip(v1, v2))   ** 1/2

