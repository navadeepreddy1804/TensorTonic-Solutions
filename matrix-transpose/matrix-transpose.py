import numpy as np

def matrix_transpose(A):
    rows=len(A)
    cols=len(A[0])
    res=[]
    for j in range(cols):
        new_row=[]
        for i in range(rows):
            new_row.append(A[i][j])
        res.append(new_row)
    return np.array(res)