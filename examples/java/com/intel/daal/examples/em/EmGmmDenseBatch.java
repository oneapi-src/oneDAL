/* file: EmGmmDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
        InitBatch initAlgorithm = new InitBatch(context, Double.class, InitMethod.defaultDense, nComponents);

        /* Set an input object for the initialization algorithm */
        initAlgorithm.input.set(InitInputId.data, input);

        InitResult initResult = initAlgorithm.compute();

        /* Create an algorithm for EM clustering */
        Batch algorithm = new Batch(context, Double.class, Method.defaultDense, nComponents);

        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);
        algorithm.input.set(InputValuesId.inputValues, initResult);

        /* Clusterize the data */
        Result result = algorithm.compute();

        NumericTable means = result.get(ResultId.means);
        NumericTable weights = result.get(ResultId.weights);

        /* Print the results */
        Service.printNumericTable("Means", means);
        Service.printNumericTable("Weights", weights);

        for (int i = 0; i < nComponents; i++) {
            NumericTable covariance = result.get(ResultCovariancesId.covariances, i);
            Service.printNumericTable("Covariance", covariance);
        }

        context.dispose();
    }
}
