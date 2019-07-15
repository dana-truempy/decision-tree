import random


def kFoldCrossValidation(data, k=5):
    """
    @param data The data set to be divided up
    @param k=5, How many crossfolds to use, default 5 
    Takes a classification data set and divides it into five, then adds one item from each class until each crossfold is full
    """
    totalObservations = len(data)
    classifierSet = {observation.classifier for observation in data}  # the set of all observations with their classifier
    stratifyByClass = {}
    for classifier in classifierSet:
        stratifyByClass[classifier] = [observation for observation in data if observation.classifier == classifier]  # stratify the data using a dictionary to hold each class name
    for key, observations in stratifyByClass.items():
        stratifyByClass[key] = {"count": len(observations), "observations": observations}
    probabilities = {classifier: stratifiedObservations["count"] / totalObservations for classifier, stratifiedObservations in stratifyByClass.items()}
    crossFolds = []
    for i in range(k):
        crossFold = []
        for classifier, probability in probabilities.items():
            crossFoldObservations = []
            while len(crossFoldObservations) / (totalObservations / k) < probability:
                if len(stratifyByClass[classifier]["observations"]) == 0:
                    break
                crossFoldObservations.append(
                    stratifyByClass[classifier]["observations"].pop(
                    random.randint(
                        0,
                        len(stratifyByClass[classifier]["observations"]) - 1
                        )
                    )
                )
            crossFold.extend(crossFoldObservations)
        crossFolds.append(crossFold)
    return crossFolds
