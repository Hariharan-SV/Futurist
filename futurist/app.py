from futurist.models.MultipleRegression import MultipleRegression
from futurist.src.model.futurist import Futurist

import warnings
warnings.filterwarnings('ignore')


Model = Futurist("D:\Library\SEM VI\ML Lab\package\\futurist\data\interim\car_dataset.csv")
Model.price_prediction(MultipleRegression())

"""
"D:\Library\SEM VI\ML Lab\package\\futurist\data\interim\car_dataset.csv"
data = load_csv("D:\Library\SEM VI\ML Lab\package\\futurist\data\external\mini_car_dataset.csv")
X,Y = pre_process_data(data)
"""