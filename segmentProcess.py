import csv
from CategoricalFeature import CategoricalFeature
from ContinuousFeature import ContinuousFeature
from Observation import Observation


def process(segmentFilename):
    """
    @param segmentFileName The path to the segmentation.data dataset
    This method takes the segmentation data and turns it into a set of Observations
    """
    with open(segmentFilename, newline='') as file:
        reader = csv.reader(file)
        segmentOut = []
        lineCount = 0

        for line in reader:
            lineCount += 1
            if lineCount <= 5:  # first five lines are a header
                continue
            segmentOut.append(line)

        observations = []
        for line in segmentOut:
            label = line[0]
            features = []
            for feature in line[1:]:  # all features are continuous
                features.append(ContinuousFeature(float(feature)))
            observations.append(Observation(label, features))

        return observations
