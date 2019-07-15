import csv
from CategoricalFeature import CategoricalFeature
from ContinuousFeature import ContinuousFeature
from Observation import Observation


def process(abaloneFilename):
    """
    @param abaloneFileName The path to the abalone.data dataset
    This method takes the abalone data and turns it into a set of Observations
    """
    with open(abaloneFilename, newline='') as file:
        reader = csv.reader(file)
        carOut = []

        for line in reader:
            className = line[-1]
            features = []
            for index, feature in enumerate(line[:-1]):
                if index == 0:
                    features.append(CategoricalFeature(feature))
                else:
                    features.append(ContinuousFeature(float(feature)))
            carOut.append(Observation(className, features))

        return(carOut)


if __name__ == "__main__":
    process("abalone.data")
