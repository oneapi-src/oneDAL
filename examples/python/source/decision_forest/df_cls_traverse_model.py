# file: df_cls_traverse_model.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#
# !  Content:
# !    Python example of decision forest classification model traversal.
# !
# !    The program trains the decision forest classification model on a training
# !    datasetFileName and prints the trained model by its depth-first traversing.
# !*****************************************************************************

#
## <a name = "DAAL-EXAMPLE-PY-DF_CLS_TRAVERSE_MODEL"></a>
## \example df_cls_traverse_model.py
#
from __future__ import print_function

from daal.algorithms import classifier
from daal.algorithms import decision_forest
import daal.algorithms.decision_forest.classification
import daal.algorithms.decision_forest.classification.training

from daal.data_management import (
    FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface, DataSourceIface, features
)

# Input data set parameters
trainDatasetFileName = "../data/batch/df_classification_train.csv"
categoricalFeaturesIndices = [2]
nFeatures = 3  # Number of features in training and testing data sets

# Decision forest parameters
nTrees = 2
minObservationsInLeafNode = 8
maxTreeDepth = 15

nClasses = 5  # Number of classes


def trainModel():

    # Create Numeric Tables for training data and dependent variables
    trainData, trainDependentVariable = loadData(trainDatasetFileName)

    # Create an algorithm object to train the decision forest classification model
    algorithm = decision_forest.classification.training.Batch(nClasses)

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainDependentVariable)

    algorithm.parameter.nTrees = nTrees
    algorithm.parameter.featuresPerNode = nFeatures
    algorithm.parameter.minObservationsInLeafNode = minObservationsInLeafNode
    algorithm.parameter.maxTreeDepth = maxTreeDepth

    # Build the decision forest classification model and return the result
    return algorithm.compute()


def loadData(fileName):

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        fileName, DataSourceIface.notAllocateNumericTable, DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and dependent variables
    data = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    dependentVar = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(data, dependentVar)

    # Retrieve the data from input file
    trainDataSource.loadDataBlock(mergedData)

    dictionary = data.getDictionary()
    for i in range(len(categoricalFeaturesIndices)):
        dictionary[categoricalFeaturesIndices[i]].featureType = features.DAAL_CATEGORICAL

    return data, dependentVar


# Visitor class implementing NodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method
class PrintNodeVisitor(classifier.TreeNodeVisitor):

    def __init__(self):
        super(PrintNodeVisitor, self).__init__()

    def onLeafNode(self, level, response):

        for i in range(level):
            print("  ", end='')
        print("Level {}, leaf node. Response value = {}".format(level, response))
        return True

    def onSplitNode(self, level, featureIndex, featureValue):

        for i in range(level):
            print("  ", end='')
        print("Level {}, split node. Feature index = {}, feature value = {:.6g}".format(level, featureIndex, featureValue))
        return True


def printModel(m):
    visitor = PrintNodeVisitor()
    print("Number of trees: {}".format(m.getNumberOfTrees()))
    for i in range(m.getNumberOfTrees()):
        print("Tree #{}".format(i))
        m.traverseDF(i, visitor)


if __name__ == "__main__":

    trainingResult = trainModel()
    printModel(trainingResult.get(classifier.training.model))
