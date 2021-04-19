import numpy as np
import pandas as pd

class MultipleRegression:
    def __init__(self) -> None:
        self.coefficients = None
        pass

    def fit(self, train_input: pd.DataFrame, train_output: pd.DataFrame) -> None:
        # x matrix
        column_cnt = len(train_input.columns) + 1
        #print(column_cnt)
        x = [[1 for _ in range(column_cnt) ] for _ in range(len(train_input)) ]
        for i in range(len(x)):
            #print(x[i][0],end=" ")
            for j in range(1,len(x[0])):
                #print(i,end=",")
                x[i][j] = train_input[train_input.columns[j-1]][train_input.index[i]]
                #print(x[i][j],end=" ")
            #print("\n")
        x = np.array(x)
        """ 
        print(x)
        print(x.transpose()) 
        """
        # x*x transpose
        pdt_x_transpose_x = np.matmul(x.transpose(),x)
        """ 
        print(pdt_x_transpose_x) 
        """
        # y matrix
        y = np.array([[i] for i in train_output])
        """ 
        print(y) 
        """
        # x transpose * y
        pdt_x_transpose_y = np.matmul(x.transpose(), y)
        """ 
        print(pdt_x_transpose_y) 
        """
        
        # coefficients
        self.coefficients = np.matmul(np.linalg.inv(pdt_x_transpose_x),pdt_x_transpose_y)
        
        #print(self.coefficients) 
        
        pass

    def predict(self, test_input: pd.DataFrame) -> None:
        outputs = []
        for _,row in test_input.iterrows():
            output = self.coefficients[0][0]
            for index,entry in enumerate(row):
                #print(self.coefficients[index+1][0]*entry,end=",")
                output += self.coefficients[index+1][0]*entry
            #print("\n")
            outputs.append(output)
        return outputs
    
    def get_params(self, **kwargs): 
        """
        params = []
        for coefficient in self.coefficients:
            params.append((coefficient[0]))
        """
        return {'sk_params':self.coefficients}