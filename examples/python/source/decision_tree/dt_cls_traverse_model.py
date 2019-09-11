# file: dt_cls_traverse_model.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

#
# !  Content:
# !    Python example of decision tree classification model traversal.
# !
# !    The program trains the decision tree classification model on a training
# !    datasetFileName and prints the trained model by its depth-first traversing.
# !*****************************************************************************

#
## <a name = "DAAL-EXAMPLE-PY-DT_CLS_TRAVERSE_MODEL"></a>
## \example dt_cls_traverse_model.py
#
from __future__ import print_function

from daal.algorithms import classifier
from daal.algorithms import decision_tree
import daal.algorithms.decision_tree.classification
import daal.algorithms.decision_tree.classification.training

from daal.data_management import (
    DataSourceIface, NumericTableIface, HomogenNumericTable, MergedNumericTable, FileDataSource
)

# Input data set parameters
trainDatasetFileName = "../data/batch/decision_tree_train.csv"
pruneDatasetFileName = "../data/batch/decision_tree_prune.csv"

nFeatures = 5
nClasses = 5


def trainModel():

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable, DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)

    # Retrieve the data from the input file
    trainDataSource.loadDataBlock(mergedData)

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the pruning input data from a .csv file
    pruneDataSource = FileDataSource(
        pruneDatasetFileName, DataSourceIface.notAllocateNumericTable, DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for pruning data and labels
    pruneData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    pruneGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    pruneMergedData = MergedNumericTable(pruneData, pruneGroundTruth)

    # Retrieve the data from the pruning input file
    pruneDataSource.loadDataBlock(pruneMergedData)

    # Create an algorithm object to train the Decision tree model
    algorithm = decision_tree.classification.training.Batch(nClasses)

    # Pass the training data set, labels, and pruning dataset with labels to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruth)
    algorithm.input.set(decision_tree.classification.training.dataForPruning, pruneData)
    algorithm.input.set(decision_tree.classification.training.labelsForPruning, pruneGroundTruth)

    # Train the Decision tree model and retrieve the results
    return algorithm.compute()


# Visitor class implementing NodeVisitor interface, prints out tree nodes of the
# model when it is called back by model traversal method
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
        print("Level {}, split node. Feature index = {}, feature value = {:.4g}".format(level, featureIndex, featureValue))

        return True


def printModel(m):
    visitor = PrintNodeVisitor()
    m.traverseDF(visitor)


if __name__ == "__main__":

    trainingResult = trainModel()
    printModel(trainingResult.get(classifier.training.model))
