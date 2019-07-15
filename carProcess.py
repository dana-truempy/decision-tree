import csv
from CategoricalFeature import CategoricalFeature
from ContinuousFeature import ContinuousFeature
from Observation import Observation


def process(carFilename):
    """
    @param carFileName The path to the car.data dataset
    @return carOut A set of Observation objects representing the data set
    This method takes the car data and turns it into a set of Observations
    """
    with open(carFilename, newline='') as file:
        reader = csv.reader(file)
        carOut = []

        for line in reader:
            className = line[-1]
            features = []
            for feature in line[:-1]:
                features.append(CategoricalFeature(feature))  # all features are categorical
            carOut.append(Observation(className, features))

        return(carOut)
