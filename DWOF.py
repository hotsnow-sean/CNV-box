from itertools import combinations
import numpy as np
from sklearn.metrics import euclidean_distances


class DWOF:
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1) -> None:
        self.n_neighbors_: int = n_neighbors if n_neighbors >= 3 else 3
        self.decision_scores_: np.ndarray = None
        self.threshold_: float = contamination
        self.labels_: np.ndarray = None

    def fit(self, data: np.ndarray):
        SIZE = len(data)
        if SIZE < 3:
            raise
        if self.n_neighbors_ > SIZE:
            self.n_neighbors_ = SIZE

        self.threshold_ *= SIZE

        distance = euclidean_distances(data)
        sort_index = np.argsort(distance, axis=1)

        # step 1: calc min distance
        all_d = distance[np.triu_indices(SIZE, 1)]
        d_min = np.amin(all_d)
        d_min = d_min if not np.isclose(d_min, 0) else (np.amax(all_d) / SIZE)
        d_min = 0.01 if np.isclose(d_min, 0) else d_min

        # step 2: calc knn avg distance of each point
        avg_d = np.zeros(SIZE)
        for i in range(SIZE):
            index = np.array(list(combinations(sort_index[i, : self.n_neighbors_], 2)))
            avg_d[i] = np.amin(distance[index[:, 0], index[:, 1]])

        # step 3: init radius and cluster size
        radius = d_min * avg_d / np.amin(avg_d)
        last_clus_size = np.full(SIZE, 1)
        self.decision_scores_ = np.zeros(SIZE)

        # step 4: update scores
        cnt = 0
        while True:
            radius *= 1.1
            clus_id = np.arange(SIZE)

            for i in range(SIZE):
                r = radius[i]
                for j in range(SIZE):
                    if distance[i, sort_index[i, j]] >= r:
                        break
                    DWOF.__union(clus_id, i, sort_index[i, j])

            for i in range(SIZE):
                DWOF.__find(clus_id, i)

            clus_cnt = np.bincount(clus_id)
            cur_clus_size = np.frompyfunc(lambda x: clus_cnt[x], 1, 1)(clus_id)

            self.decision_scores_ = (
                self.decision_scores_ + (last_clus_size - 1) / cur_clus_size
            )
            last_clus_size = cur_clus_size
            cnt += 1

            if cnt > 2 and (
                np.amax(clus_cnt) >= SIZE - self.threshold_
                or (SIZE > 100 and np.count_nonzero(clus_cnt) < 5)
            ):
                break

    def __find(arr: np.ndarray, id: int) -> int:
        p = id
        while p != arr[p]:
            p = arr[p]
        while id != arr[id]:
            tmp = id
            id = arr[id]
            arr[tmp] = p
        return p

    def __union(arr: np.ndarray, a: int, b: int):
        ac = DWOF.__find(arr, a)
        bc = DWOF.__find(arr, b)
        arr[ac] = bc
