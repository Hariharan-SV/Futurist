from futurist.models.MultipleRegression import MultipleRegression
from futurist.src.model.futurist import Futurist
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

path = Path(__file__).parent / "data/interim/car_dataset.csv"
Model = Futurist(path,deep_clean=True)
Model.data_characterstics(["name","engine","max_power","torque","mileage"])
Model.price_prediction(MultipleRegression(),print_report=True)

"""

path = Path(__file__).parent / "data/interim/car_dataset.csv"
Model = Futurist(path,deep_clean=True)
Model.data_characterstics(["name","engine","max_power","torque","mileage"])
Model.price_prediction(MultipleRegression(),print_report=True)

path = Path(__file__).parent / "data/external/mini_car_dataset.csv"
Model = Futurist(path)
Model.data_characterstics(["Car_Name"])
Model.price_prediction(MultipleRegression(),print_report=True)

"""