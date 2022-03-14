/* file: CorCSRBatch.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 //     Java example of correlation matrix computation in the batch
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-CORRELATIONCSRBATCH">
 * @example CorCSRBatch.java
 */

package com.intel.daal.examples.covariance;

import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class CorCSRBatch {
    /*
     * Input data set parameters
     * Input matrix is stored in the compressed sparse row format with one-based indexing
     */
    private static final String datasetFileName = "../data/batch/covcormoments_csr.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read a data set from a file and create a numeric table for storing the input data */
        CSRNumericTable dataTable = Service.createSparseTable(context, datasetFileName);

        /* Create an algorithm to compute a correlation matrix using the default method */
        Batch alg = new Batch(context, Float.class, Method.fastCSR);
        alg.input.set(InputId.data, dataTable);

        /* Set the parameter to choose the type of the output matrix */
        alg.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix);

        /* Compute the correlation matrix */
        Result res = alg.compute();

        HomogenNumericTable correlation = (HomogenNumericTable) res.get(ResultId.correlation);
        HomogenNumericTable mean = (HomogenNumericTable) res.get(ResultId.mean);

        Service.printNumericTable("Correlation matrix (upper left square 10*10) :", correlation, 10, 10);
        Service.printNumericTable("Mean vector:", mean, 1, 10);

        context.dispose();
    }
}
