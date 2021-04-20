import numpy as np
import pandas as pd

class MultipleRegression:
    def __init__(self) -> None:
        self.coefficients = None
        pass

    def fit(self, train_input: pd.DataFrame, train_output: pd.DataFrame) -> None:
        # x matrix
        x = [[1 for _ in range( len(train_input.columns)+1 ) ] for _ in range(len(train_input)) ]
        for i in range(len(x)):
            for j in range(1,len(x[0])):
                x[i][j] = train_input[train_input.columns[j-1]][train_input.index[i]]
        x = np.array(x)
        # x*x transpose
        pdt_x_transpose_x = np.matmul(x.transpose(),x)
        # y matrix
        y = np.array([[i] for i in train_output])
        # x transpose * y
        pdt_x_transpose_y = np.matmul(x.transpose(), y)
        # coefficients
        self.coefficients = np.matmul(np.linalg.inv(pdt_x_transpose_x),pdt_x_transpose_y) 
        pass

    def predict(self, test_input: pd.DataFrame) -> None:
        # predicted output list
        outputs = []
        for _,row in test_input.iterrows():
            # bias
            output = self.coefficients[0][0]
            for index,entry in enumerate(row):
                # product of feature coefficient and value
                output += self.coefficients[index+1][0]*entry
            outputs.append(output)
        return outputs
    