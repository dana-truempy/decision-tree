class Feature():

    def __init__(self, type, value):
        """
        @param type Categorical or Continuous, passed in from the CategoricalFeature and ContinuousFeature classes
        @param value The numerical or string value of the feature 
        Instantiates a new feature with the type (categorical or continuous) and the value of the feature
        """
        self._type = type
        self._value = value

    def isComparable(self, otherFeature):
        """
        @param otherFeature: Feature class to determine comparability with 
        Check to see if this feature is comparable with another feature
        """
        return self._type == otherFeature._type

