class Node():

    def __init__(self, data, children=None, splitType=None, splitFeature=None, splitValue=None, classifier=None):
        """
        @param self
        @param data The data that the node will hold - corresponds to feature and value
        @param children The other Nodes that the Node points to
        @param splitType Whether the split is on values greater than, less than, or equal to the feature value
        @param splitFeature The index number of the feature that is being used for the rule/to split
        @param splitValue The feature value that the rule applies to
        @param classifier The majority class of the data in the node
        Class constructor for a Node object
        """
        self.data = data
        self.children = children
        self.splitType = splitType
        self.splitFeature = splitFeature
        self.splitValue = splitValue
        self.classifier = classifier

    def isLeaf(self):
        """
        @param self
        @return boolean True if the Node has no children
        This method tests if a Node is a leaf node, ie if it has children
        """
        return self.children == None

    def checkNode(self, observation):
        """
        @param self
        @param observation An Observation object containing features that can be tested against the node's rule
        @return If the node's rule applies to the observation
        This method checks an observation against the rule of the node - if the observation matches the rule,
        it visits this node and continues down the tree.
        """
        if self.splitType == "equals":
            return observation.features[self.splitFeature]._value == self.splitValue

        elif self.splitType == "less than":
            return observation.features[self.splitFeature]._value < self.splitValue

        elif self.splitType == "greater than":
            return observation.features[self.splitFeature]._value >= self.splitValue
