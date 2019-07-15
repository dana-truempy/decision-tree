class Observation:
    """
    This class creates an Observation instance, which is an object with a Class/Label and a set of Features associated.
    """

    def __init__(self, classifier, features):
        """
        @param classifier The class/label or regression value for an observation
        @param features A list of Feature objects that represent the values of each feature in the observation
        The init method instantiates the Observation, giving it a class and a set of Features.
        """
        if features is None:  # each observation needs features, but a testing set with unknown class can be used
            raise ValueError("Features cannot be null")
        self.features = features
        self.classifier = classifier

