from copy import copy
import inspect
import math
from operator import itemgetter
import sys

from CrossValidation import kFoldCrossValidation
from Feature import Feature
from Node import Node
from Observation import Observation
import abaloneProcess
import carProcess
import segmentProcess

theta = 0.3  # error threshold for early stopping - theta for cutoff of some low value of gain


def main():
    """
    This is a main method to call all of the other scripts, takes one of the three classification data sets for this assignment 
    and separates into validation, testing and training sets, then generates, prunes, and tests a decision tree 
    """

    if 'abalone' in sys.argv[1].lower():
        data = abaloneProcess.process(sys.argv[1])

    elif 'car' in sys.argv[1].lower():
        data = carProcess.process(sys.argv[1])

    elif 'segment' in sys.argv[1].lower():
        data = segmentProcess.process(sys.argv[1])

    validationSet = []
    trainingSet = []

    for index, line in enumerate(data):  # takes every 10th element in the data and pulls it for the validation set, otherwise it's training/testing
        if index % 10 == 0:
            validationSet.append(line)
        else:
            trainingSet.append(line)

    crossFolds = kFoldCrossValidation(trainingSet)  # get the five stratified crossfolds
    train = []
    accuracies = []
    pruneAccuracies = []

    for i in range(len(crossFolds)):
        print("Testing with crossfold number " + str(i + 1))
        train = []
        for crossFold in crossFolds[:i] + crossFolds[i + 1:]:  # use all crossfolds but the test one as training set
            train.extend(crossFold)

        features = [index for index in range(len(train[0].features))]  # start with the complete set of features (index numbers of the features)
        root = Node(train)  # the root starts as just the full training set

        generateTree(root, features)  # make the tree first

        baseAccuracy = totalError(root, crossFolds[i])  # the base accuracy is just accuracy measured before any pruning takes place
        accuracies.append(baseAccuracy)  # compile the accuracies of each cross-validation run

        pruneImprovement = 100

        while pruneImprovement > 0.1:  # can set the value that pruning should improve the data set
            pruneImprovement, pruneNode = pruneTree(root, validationSet)  # prune the data set using the validation set
            if pruneNode != None:  # if the method returns a node to be pruned then prune it
                pruneNode.children = None

        pruneAccuracy = totalError(root, crossFolds[i])  # measure how well the pruned tree does on the testing data
        pruneAccuracies.append(pruneAccuracy)  # put together all of the pruned accuracies

        print("Accuracy (no pruning): " + str(baseAccuracy))
        print("Accuracy (with pruning): " + str(pruneAccuracy))

    totalBase = sum(accuracies) / len(accuracies)
    totalPrune = sum(pruneAccuracies) / len(pruneAccuracies)
    print("Average accuracy over five-fold cross-validation: ")
    print("Without pruning: " + str(totalBase))
    print("With pruning: " + str(totalPrune))


def entropy(node):
    """
    @param node The method takes a single node as input
    @return ent The calculated entropy of the node
    This method uses the data of a node to find and return its entropy 
    """
    if len(node.data) == 0:  # if the node is empty then it isn't a reasonable solution
        ent = 100
    else:
        classProbs = {}  # count how many observations in the node's data correspond to each class
        for observation in node.data:
            if observation.classifier not in classProbs:
                classProbs[observation.classifier] = 1
            else:
                classProbs[observation.classifier] += 1

        for prob in classProbs:
            classProbs[prob] /= len(node.data)  # divide the class probability by the number of observations

        ent = 0
        for prob in classProbs:  # entropy is the number belonging to a class/total number multiplied by log2 of that same value, all summed together
            ent += (-1 * classProbs[prob] * math.log2(classProbs[prob]))

    return ent


def gainRatio(parent, children):
    """
    @param parent The parent node for which to calculate the split's gain ratio
    @param children All of the children of the parent that can be used to calculate the purity of the split
    @return gainRatio The ratio of information gain over intrinsic value of a feature
    This method takes a parent node and its children and calculates the gain ratio for splitting that parent  
    """
    avgEntropy = 0
    for child in children:  # the entropy of each child is multiplied by the size of its data set compared to the parent's
        childEntropy = entropy(child) * (len(child.data) / len(parent.data))
        avgEntropy += childEntropy

    infoGain = (entropy(parent) - avgEntropy)  # information gain is the parent node's entropy minus the weighted average entropy of each child

    intrinsicValue = 0

    for child in children:  # find the intrinsic value
        if len(child.data) == 0:  # can't take log of 0, returns error - but 0 times log anything would be 0 so its value is 0
            intrinsicValue += 0
        else:
            featureProbability = len(child.data) / len(parent.data)
            # the entropy calculation for the number of observations with each feature value over the total observations
            intrinsicValue += (-1 * featureProbability * math.log2(featureProbability))

    if intrinsicValue == 0:  # if there are no children or any potential split would have children with no useful data, creating this split is useless
        return 0
    else:
        gainRatio = infoGain / intrinsicValue  # if the split has a real valued return then return the gain ratio
        return gainRatio


def split(parent, currFeatures):
    """
    @param parent The node which is being split 
    @param currFeatures The current list of features still being considered 
    @return currFeatures The list of features after the data has been split on the best one
    @return maxGain The highest gain that was gotten from the split
    @return bestChildren The children to be created from the parent node split
    """
    maxGain = -10000
    if len(currFeatures) == 0:  # if all the features have been used to split, the tree can't be split any more
        return(None, 0, None)

    for feature in currFeatures:
        if parent.data[0].features[feature]._type == "categorical":  # categorical and continuous features are treated differently
            splits = {}
            children = []

            for observation in parent.data:
                if observation.features[feature]._value not in splits:  # create a dictionary key for each possible value of the feature
                    splits[observation.features[feature]._value] = []
                splits[observation.features[feature]._value].append(observation)  # each observation with that feature value is added to the data

            for featureValue in splits:  # create children using all data containing a given feature value and the rule to be used to split that data
                newNode = Node(data=splits[featureValue], splitType="equals", splitFeature=feature, splitValue=featureValue)
                children.append(newNode)

            gain = gainRatio(parent, children)  # calculate the information gain ratio for this potential split

            if gain > maxGain:  # if the gain is the best, set the gain, feature, and children as the best
                maxGain = gain
                bestChildren = children
                bestFeature = feature

        elif parent.data[0].features[feature]._type == "continuous":
            featureValues = []
            children = []

            for observation in parent.data:
                featureValues.append([observation.features[feature]._value, observation.classifier])  # list of the feature's value and the class of the observation

            featureValues.sort(key=itemgetter(0))  # sort the list of values on the feature value as seen in the discussion board

            splitValue = 0
            splits = []

            for index, value in enumerate(featureValues):
                if index == (len(featureValues) - 1):
                    break
                elif value[1] == featureValues[index + 1][1]:  # if the class of two of the sorted values falls together, skip that value
                    continue
                else:
                    splitValue = ((value[0] + featureValues[index + 1][0]) / 2)  # split the data by the mean of any two values that have consecutive feature values
                    if splitValue not in splits:
                        splits.append(splitValue)

            if len(splits) == 0:  # if every class in the node is the same then the node is perfectly sorted
                bestFeature = None
                maxGain = 0
                return(bestFeature, maxGain, None)

            else:
                for value in splits:
                    minSplit = []
                    maxSplit = []
                    for observation in parent.data:  # split the data into two sections: one with values lower than the split and one with higher
                        if observation.features[feature]._value < value:
                            minSplit.append(observation)
                        else:
                            maxSplit.append(observation)

                    children = []  # make child nodes for each split
                    children.append(Node(minSplit, None, "less than", feature, value))
                    children.append(Node(maxSplit, None, "greater than", feature, value))

                    gain = gainRatio(parent, children)

                    if gain > maxGain:  # if gain ratio is better than other features, set gain, children, feature as best
                        maxGain = gain
                        bestChildren = copy(children)
                        bestFeature = feature

    if parent.data[0].features[bestFeature]._type == "categorical":  # continuous features can be split on more than once, categorical shouldn't be
        currFeatures.remove(bestFeature)

    return (currFeatures, maxGain, bestChildren)


def generateTree(currNode, currFeatures):
    """
    @param currNode The current node that is being traversed and split
    @param currFeatures The list of features still available to be split on
    This method goes through the tree, splitting it until splitting no longer has positive gain.
    It's a recursive method that grows out to each leaf before returning up 
    """
    splitFeature = split(currNode, currFeatures)  # start by splitting the current node and seeing if its gain is positive
    currNode.classifier = getClass(currNode.data)[0]  # add a classifier to every node for pruning later

    if splitFeature[1] <= theta:  # if the split does not improve the gain ratio
        return

    else:
        currFeatures = splitFeature[0]  # the features list with (potentially) the best feature from split removed
        currNode.children = splitFeature[2]  # the best children returned by splitting
        for child in currNode.children:
            generateTree(child, currFeatures)  # recursively go through each child and split until splitting is no longer valuable


def getClass(data):
    """
    @param data The data set held by a node 
    @return The majority class
    Takes a node's data set and finds which class is the most frequent
    """
    classCounts = {}
    for observation in data:
        if observation.classifier not in classCounts:  # dictionary of counts of each class
            classCounts[observation.classifier] = 1
        else:
            classCounts[observation.classifier] += 1

    return max(classCounts.items(), key=lambda x: x[1])  # compare the lengths of the class counts and return the class with the most observations


def testTree(node, observation):
    """
    @param node A single Node object from a tree
    @param observation One observation from a data set to be classified by the tree
    @return node.classifier The majority class of the leaf node reached by the method
    This method classifies a single observation using the tree from generateTree or pruneTree
    """
    if node.isLeaf():  # recursively goes through the tree until hitting a leaf and using it to classify the observation
        return node.classifier

    else:
        for child in node.children:
            if child.checkNode(observation):  # checks the type of comparison to be made and returns if the rule is true for that node
                return testTree(child, observation)


def totalError(root, observations):
    """
    @param root The root of the current tree
    @param observations The complete data set to be analyzed 
    @return accuracy The number of mistakes out of the total over the total
    This method tests a testing set on a generated tree and returns the accuracy rate for that tree on that data set
    """
    mistakes = 0

    for obs in observations:
        test = testTree(root, obs)

        if test != obs.classifier:  # if the tree classifies incorrectly, list as mistake
            mistakes += 1

    accuracy = (len(observations) - mistakes) / len(observations)
    return(accuracy)


def getLeaves(node, pruneNodes):
    """
    @param node Takes a Node object
    @param pruneNodes Iteratively adds to the list of leaves to be analyzed for pruning
    @return pruneNodes When the method reaches the parent of a leaf, it adds that parent to the list to be pruned
    This method takes a root node and recursively traverses the tree, marking down each node that comes directly before leaf nodes
    """
    if node.children == None:  # if the tree is just the root node and hasn't split, then pruning is impossible
        return []
    else:
        for child in node.children:  # go through the node's children to see if they are leaves
            if child.isLeaf():
                if node not in pruneNodes:
                    pruneNodes.append(node)  # if the node is a parent to leaves, add it to the potential pruning list
                return(pruneNodes)
            else:
                return(getLeaves(child, pruneNodes))  # otherwise, continue traversing the tree


def pruneTree(root, validationSet):
    """
    @param root The root of the tree to be pruned
    @param validationSet The validation set to be used for pruning the tree
    @return bestImprovement How much better the pruned tree did on the validation set than the un-pruned tree
    @return pruneNode The Node object to be pruned in the main method
    This method repeatedly goes through the leaves of a tree and prunes them one by one, testing them to see if they perform better than 
    a tree that has not been pruned.
    """
    baseAccuracy = totalError(root, validationSet)  # get the accuracy of the tree before pruning takes place
    bestImprovement = -1000
    leafNodes = getLeaves(root, [])  # get all of those nodes that have children that are leaves

    if len(leafNodes) == 0:  # if the tree is just the root, then pruning cannot take place
        return 0, None
    else:
        for node in leafNodes:
            children = node.children  # for each node, save the children of the node, then remove them and test the tree without them
            node.children = None
            accuracy = totalError(root, validationSet)
            improvement = baseAccuracy - accuracy  # the improvement is how much better the pruned tree did than the unpruned tree
            node.children = children  # reassign the pointers to the children after testing
            if improvement > bestImprovement:  # if the tree improves, save the best leaves to prune
                bestImprovement = improvement
                pruneNode = node

    return bestImprovement, pruneNode


if __name__ == "__main__":
    main()
