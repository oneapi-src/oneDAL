/* file: assoc_rules_apriori_batch.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!    C++ example of association rules mining
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-APRIORI_BATCH"></a>
 * \example assoc_rules_apriori_batch.cpp
 */

#include "daal.h"
#include "service.h"
using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName = "../data/batch/apriori.csv";

/* Apriori algorithm parameters */
const double minSupport     = 0.001;    /* Minimum support */
const double minConfidence  = 0.7;      /* Minimum confidence */

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to mine association rules using the Apriori method */
    association_rules::Batch<> algorithm;

    /* Set the input object for the algorithm */
    algorithm.input.set(association_rules::data, dataSource.getNumericTable());

    /* Set the Apriori algorithm parameters */
    algorithm.parameter.minSupport = minSupport;
    algorithm.parameter.minConfidence = minConfidence;

    /* Find large item sets and construct association rules */
    algorithm.compute();

    /* Get computed results of the Apriori algorithm */
    association_rules::ResultPtr res = algorithm.getResult();

    /* Print the large item sets */
    printAprioriItemsets(res->get(association_rules::largeItemsets),
                         res->get(association_rules::largeItemsetsSupport));

    /* Print the association rules */
    printAprioriRules(res->get(association_rules::antecedentItemsets),
                      res->get(association_rules::consequentItemsets),
                      res->get(association_rules::confidence));

    return 0;
}
