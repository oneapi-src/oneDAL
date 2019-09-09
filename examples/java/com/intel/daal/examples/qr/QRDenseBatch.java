/* file: QRDenseBatch.java */
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
 //     Java example of computing QR decomposition in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-QRBATCH">
 * @example QRDenseBatch.java
 */

package com.intel.daal.examples.qr;

import com.intel.daal.algorithms.qr.Batch;
import com.intel.daal.algorithms.qr.InputId;
import com.intel.daal.algorithms.qr.Method;
import com.intel.daal.algorithms.qr.Result;
import com.intel.daal.algorithms.qr.ResultId;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class QRDenseBatch {

    /* Input data set parameters */
    private static final String dataset  = "../data/batch/qr.csv";
    private static final int    nVectors = 100;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from an input file */
        dataSource.loadDataBlock();

        NumericTable input = dataSource.getNumericTable();

        /* Create an algorithm to compute QR decomposition */
        Batch qrAlgorithm = new Batch(context, Float.class, Method.defaultDense);
        qrAlgorithm.input.set(InputId.data, input);

        /* Compute QR decomposition */
        Result res = qrAlgorithm.compute();

        NumericTable matrixQ = res.get(ResultId.matrixQ);
        NumericTable matrixR = res.get(ResultId.matrixR);

        /* Print the results */
        Service.printNumericTable("Orthogonal matrix Q:", matrixQ, 10);
        Service.printNumericTable("Triangular matrix R:", matrixR);

        context.dispose();
    }
}
