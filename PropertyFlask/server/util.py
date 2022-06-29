#To ignore Sklearn Warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pickle
import json
import numpy as np


__locations =  None
__data_columns = None
__model = None

def load_saved_artifacts():
    print("Loading saved artifacts... Start")
    global __data_columns
    global __locations
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    if __model is None:
        with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
            __model = pickle.load(f)
    print("Loading saved artifacts... Done")

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)



def get_location_names():
    global __locations
    return __locations

def get_data_columns():
    global __data_columns
    return __data_columns


load_saved_artifacts()
# get_estimated_price(location,sqft,bhk,bath)

# if __name__ == '__main__':
#     load_saved_artifacts()
#     print(get_location_names())
    # print(get_data_columns())

    # print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    # print(get_estimated_price('Kalhalli', 1000, 2, 2))
    # print(get_estimated_price('Ejipura', 1000, 2, 2))