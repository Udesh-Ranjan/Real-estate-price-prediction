import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_location_names():
    return __locations


def get_estimated_price(location, sqft, bhk, bath):
    loc_index = -1
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        print('error get_estimated_price', location, sqft, bhk, bath)
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    global __data_columns
    global __model
    global __locations
    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)


def main():
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 2))


if __name__ == '__main__':
    main()