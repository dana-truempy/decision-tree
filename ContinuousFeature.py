from Feature import Feature


class ContinuousFeature(Feature):
    """
    This class calls the parent Feature class and returns the type and distance metric for a continuous feature
    """

    def __init__(self, value):
        """
        @param value The value that the feature takes for a given observation
        Instantiates a new Feature of "continuous" type
        """
        super().__init__('continuous', value)
