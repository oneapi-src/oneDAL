# file: assoc_rules_apriori_batch.py
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
