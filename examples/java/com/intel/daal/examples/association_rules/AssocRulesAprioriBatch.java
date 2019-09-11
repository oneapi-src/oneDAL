/* file: AssocRulesAprioriBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 //  Content:
 //     Java example of association rules mining
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-APRIORIBATCH">
 * @example AssocRulesAprioriBatch.java
 */

package com.intel.daal.examples.association_rules;

import com.intel.daal.algorithms.association_rules.*;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class AssocRulesAprioriBatch {
    /* Input data set parameters */
    private static final String dataset  = "../data/batch/apriori.csv";

    /* Apriori algorithm parameters */
    private static final double minSupport    = 0.001; /* Minimum support */
    private static final double minConfidence = 0.7;   /* Minimum confidence */

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm to mine association rules using the Apriori method */
        Batch alg = new Batch(context, Float.class, Method.apriori);

        /* Set an input object for the algorithm */
        NumericTable input = dataSource.getNumericTable();
        alg.input.set(InputId.data, input);

        /* Set Apriori algorithm parameters */
        alg.parameter.setMinSupport(minSupport);
        alg.parameter.setMinConfidence(minConfidence);

        /* Find large item sets and construct association rules */
        Result res = alg.compute();

        HomogenNumericTable largeItemsets = (HomogenNumericTable) res.get(ResultId.largeItemsets);
        HomogenNumericTable largeItemsetsSupport = (HomogenNumericTable) res.get(ResultId.largeItemsetsSupport);

        /* Print the large item sets */
        Service.printAprioriItemsets(largeItemsets, largeItemsetsSupport);

        HomogenNumericTable antecedentItemsets = (HomogenNumericTable) res.get(ResultId.antecedentItemsets);
        HomogenNumericTable consequentItemsets = (HomogenNumericTable) res.get(ResultId.consequentItemsets);
        HomogenNumericTable confidence = (HomogenNumericTable) res.get(ResultId.confidence);

        /* Print the association rules */
        Service.printAprioriRules(antecedentItemsets, consequentItemsets, confidence);

        context.dispose();
    }
}
