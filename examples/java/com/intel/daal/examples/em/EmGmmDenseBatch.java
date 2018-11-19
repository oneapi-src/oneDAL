/* file: EmGmmDenseBatch.java */
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
 //  Content:
 //     Java example of the expectation-maximization (EM) algorithm for the
 //     Gaussian mixture model (GMM)
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-EMGMMBATCH">
 * @example EmGmmDenseBatch.java
 */

package com.intel.daal.examples.em;

import com.intel.daal.algorithms.em_gmm.*;
import com.intel.daal.algorithms.em_gmm.init.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class EmGmmDenseBatch {
    /* Input data set parameters */
    private static final String dataset       = "../data/batch/em_gmm.csv";
    private static final int    nComponents   = 2;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();
        NumericTable input = dataSource.getNumericTable();

        /* Create an algorithm to initialize the EM algorithm for the GMM */
        InitBatch initAlgorithm = new InitBatch(context, Float.class, InitMethod.defaultDense, nComponents);

        /* Set an input object for the initialization algorithm */
        initAlgorithm.input.set(InitInputId.data, input);

        InitResult initResult = initAlgorithm.compute();

        /* Create an algorithm for EM clustering */
        Batch algorithm = new Batch(context, Float.class, Method.defaultDense, nComponents);

        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);
        algorithm.input.set(InputValuesId.inputValues, initResult);

        /* Clusterize the data */
        Result result = algorithm.compute();

        NumericTable means = result.get(ResultId.means);
        NumericTable weights = result.get(ResultId.weights);

        /* Print the results */
        Service.printNumericTable("Weights", weights);
        Service.printNumericTable("Means", means);

        for (int i = 0; i < nComponents; i++) {
            NumericTable covariance = result.get(ResultCovariancesId.covariances, i);
            Service.printNumericTable("Covariance", covariance);
        }

        context.dispose();
    }
}
