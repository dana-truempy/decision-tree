from Feature import Feature


class CategoricalFeature(Feature):
    """
    This class calls the parent Feature class and returns the type and distance metric for a categorical feature
    """

    def __init__(self, value):
        """
        @param value The value that the feature takes for a given observation
        Instantiates a new Feature of "categorical" type
        """
        super().__init__('categorical', value)
