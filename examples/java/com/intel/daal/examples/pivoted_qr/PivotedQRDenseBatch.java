/* file: PivotedQRDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 //     Java example of computing pivoted QR decomposition
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PIVOTEDQRBATCH">
 * @example PivotedQRDenseBatch.java
 */

package com.intel.daal.examples.pivoted_qr;

import com.intel.daal.algorithms.pivoted_qr.Batch;
import com.intel.daal.algorithms.pivoted_qr.InputId;
import com.intel.daal.algorithms.pivoted_qr.Method;
import com.intel.daal.algorithms.pivoted_qr.Result;
import com.intel.daal.algorithms.pivoted_qr.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class PivotedQRDenseBatch {

    /* Input data set parameters */
    private static final String dataset  = "../data/batch/qr.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();

        NumericTable input = dataSource.getNumericTable();

        /* Create an algorithm to compute pivoted QR decomposition */
        Batch pivotedQRAlgorithm = new Batch(context, Float.class, Method.defaultDense);
        pivotedQRAlgorithm.input.set(InputId.data, input);

        /* Compute pivoted QR decomposition */
        Result res = pivotedQRAlgorithm.compute();

        NumericTable matrixQ = res.get(ResultId.matrixQ);
        NumericTable matrixR = res.get(ResultId.matrixR);
        NumericTable permutationMatrix = res.get(ResultId.permutationMatrix);

        /* Print the results */
        printResults(input, matrixQ, matrixR, permutationMatrix);

        context.dispose();
    }

    private static void printResults(NumericTable data, NumericTable Q, NumericTable R, NumericTable P) {
        Service.printNumericTable("Orthogonal matrix Q:", Q, 10);
        Service.printNumericTable("Triangular matrix R:", R);
        Service.printNumericTable("Permutation matrix P:", P);
    }
}
