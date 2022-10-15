class FuzzyCMeansMembership:

    def memberships(distances, m):
        memberships = []
        for distance_j in distances:
            # To avoid division by 0
            sum_of_distances = 2.2250738585072014e-308
            for distance_k in distances:
                if distance_k != 0:
                    sum_of_distances += pow((distance_j / distance_k), 2. / (m - 1.))
            memberships.append(1.0 / sum_of_distances)
        return memberships
