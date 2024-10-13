# TODO: Move to SDK
import numpy as np

from uint256 import XoShiRo256PlusPlus, Hash

class Matrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.uint16)

    @classmethod
    def rand_matrix_no_rank_check(cls, generator):
        def generate_row():
            val = 0
            row = np.zeros(64, dtype=np.uint16)
            for j in range(64):
                shift = j % 16
                if shift == 0:
                    val = generator.u64()
                row[j] = (val >> (4 * shift)) & 0x0F
            return row

        return cls(np.array([generate_row() for _ in range(64)]))

    def convert_to_float(self):
        return self.data.astype(np.float64)

    def compute_rank(self):
        EPS = 1e-9
        mat_float = self.convert_to_float()
        rank = 0
        row_selected = [False] * 64

        for i in range(64):
            j = 0
            while j < 64:
                if not row_selected[j] and abs(mat_float[j][i]) > EPS:
                    break
                j += 1
            
            if j != 64:
                rank += 1
                row_selected[j] = True
                mat_float[j, i+1:] /= mat_float[j, i]
                
                for k in range(64):
                    if k != j and abs(mat_float[k, i]) > EPS:
                        mat_float[k, i+1:] -= mat_float[j, i+1:] * mat_float[k, i]

        return rank

    @classmethod
    def generate(cls, hash):
        generator = XoShiRo256PlusPlus(Hash(hash))
        while True:
            mat = cls.rand_matrix_no_rank_check(generator)
            # This gives 100.00 H/s
            # if mat.compute_rank() == 64: # TODO: Replace with np.linalg.matrix_rank(mat.data)
            # This gives 377.00 H/s
            if np.linalg.matrix_rank(mat.data) == 64:
                return mat