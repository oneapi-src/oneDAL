/* file: QuantilesDenseBatch.java */
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
 //     Java example of computing quantiles
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-QUANTILESBATCH">
 * @example QuantilesDenseBatch.java
 */

package com.intel.daal.examples.quantiles;

import com.intel.daal.algorithms.quantiles.Batch;
import com.intel.daal.algorithms.quantiles.InputId;
import com.intel.daal.algorithms.quantiles.Method;
import com.intel.daal.algorithms.quantiles.Result;
import com.intel.daal.algorithms.quantiles.ResultId;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class QuantilesDenseBatch {
    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/quantiles.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm to compute quantiles using the single-pass method */
        Batch algorithm = new Batch(context, Float.class, Method.defaultDense);
        algorithm.input.set(InputId.data, dataSource.getNumericTable());

        /* Compute quantiles */
        Result res = algorithm.compute();

        Service.printNumericTable("Quantiles:", res.get(ResultId.quantiles));

        context.dispose();
    }
}
