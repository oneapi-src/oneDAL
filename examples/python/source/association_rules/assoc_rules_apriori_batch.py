# file: assoc_rules_apriori_batch.py
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

## <a name="DAAL-EXAMPLE-PY-APRIORI_BATCH"></a>
## \example assoc_rules_apriori_batch.py

import os
import sys

from daal.algorithms import association_rules
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printAprioriItemsets, printAprioriRules

#  Input data set parameters
datasetFileName = os.path.join('..','data','batch','apriori.csv')

#  Apriori algorithm parameters
minSupport = 0.001
minConfidence = 0.7

#  Initialize FileDataSource_CSVFeatureManager to retrieve the input data from a .csv file
dataSource = FileDataSource(
    datasetFileName, DataSourceIface.doAllocateNumericTable, DataSourceIface.doDictionaryFromContext
)

#  Retrieve the data from the input file
dataSource.loadDataBlock()

#  Create an algorithm to mine association rules using the Apriori method
alg = association_rules.Batch()
alg.input.set(association_rules.data, dataSource.getNumericTable())
alg.parameter.minSupport = minSupport
alg.parameter.minConfidence = minConfidence

#  Find large item sets and construct association rules
res = alg.compute()

#  Get computed results of the Apriori algorithm
nt1 = res.get(association_rules.largeItemsets)
nt2 = res.get(association_rules.largeItemsetsSupport)

nt3 = res.get(association_rules.antecedentItemsets)
nt4 = res.get(association_rules.consequentItemsets)
nt5 = res.get(association_rules.confidence)

printAprioriItemsets(nt1, nt2)
printAprioriRules(nt3, nt4, nt5)
