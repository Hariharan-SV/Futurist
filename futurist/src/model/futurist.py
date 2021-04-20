# Script to train and predict

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from futurist.src.features.pre_process import pre_process, pre_process_data

class Futurist:

    def __init__(self,path,test_size=0.33,random_state=5,deep_clean=False) -> None:
        """
        """
        self.path = path
        self.CV = []
        self.R2_train = []
        self.R2_test = []
        self.split_data(test_size,random_state,deep_clean)
        pass

    def load_csv(self) -> pd.DataFrame:
        """
        """
        file = pd.read_csv(self.path)
        df = pd.DataFrame(file)
        self.data = df.apply(lambda x: pd.Series(x.dropna().values))
        pass

    def data_characterstics(self,ignorable_columns=[]):
        """
        """
        final_df = self.data.dropna()
        # Heatmap
        sns.heatmap(final_df.corr(method ='kendall'), annot=True, cmap="RdBu")
        plt.show()
        # Plots
        for column in final_df.columns:
            if column in ignorable_columns:
                continue
            if final_df.dtypes[column] == object:
                sns.countplot(x=column, data=final_df)
            else:
                sns.boxplot(x=column, data=final_df)
            plt.title(column)
            plt.show()
        pass

    def split_data(self, test_size,random_state,deep_clean) -> None:
        """
        """
        self.load_csv()
        if deep_clean:
            X,Y = pre_process(self.data)
        else:
            X,Y = pre_process_data(self.data)
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,Y,test_size=test_size,random_state=random_state)
        pass

    def price_prediction(self,model,print_report=False):
        """
        """

        # Training model
        model.fit(self.X_train,self.y_train)

        # R2 score of train set
        y_pred_train = model.predict(self.X_train)
        self.R2_train_model = r2_score(self.y_train,y_pred_train)
        self.R2_train.append(round(self.R2_train_model,2))

        # R2 score of test set
        y_pred_test = model.predict(self.X_test)
        self.R2_test_model = r2_score(self.y_test,y_pred_test)
        self.R2_test.append(round(self.R2_test_model,2))

        if not print_report:
            return
        
        # Report
        print("x train: ",self.X_train.shape)
        print("x test: ",self.X_test.shape)
        print("y train: ",self.y_train.shape)
        print("y test: ",self.y_test.shape)

        # Printing results
        print("Train R2-score\t:",round(self.R2_train_model,2))
        print("Test R2-score\t:",round(self.R2_test_model,2))

        # Plotting Graphs 
        # Residual Plot of train data
        fig, ax = plt.subplots(1,2,figsize = (10,4))
        ax[0].set_title('Residual Plot of Train samples')
        sns.distplot((self.y_train-y_pred_train),hist = False,ax = ax[0])
        ax[0].set_xlabel('y_train - y_pred_train')
        
        # self.Y_test vs self.Y_train scatter plot
        ax[1].set_title('y_test vs y_pred_test')
        ax[1].scatter(x = self.y_test, y = y_pred_test)
        ax[1].set_xlabel('y_test')
        ax[1].set_ylabel('y_pred_test')
        
        plt.show()
        pass


