/* file: SortingDenseBatch.java */
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
 //     Java example of sorting the observations matrix
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SORTINGBATCH">
 * @example SortingDenseBatch.java
 */

package com.intel.daal.examples.sorting;

import com.intel.daal.algorithms.sorting.Batch;
import com.intel.daal.algorithms.sorting.InputId;
import com.intel.daal.algorithms.sorting.Method;
import com.intel.daal.algorithms.sorting.Result;
import com.intel.daal.algorithms.sorting.ResultId;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class SortingDenseBatch {
    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/sorting.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                                                       DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                       DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Print the input observations matrix */
        Service.printNumericTable("Initial matrix of observations:", dataSource.getNumericTable());

        /* Create an algorithm to sort data  */
        Batch algorithm = new Batch(context, Float.class, Method.defaultDense);
        algorithm.input.set(InputId.data, dataSource.getNumericTable());

        /* Sort data observations */
        Result res = algorithm.compute();

        Service.printNumericTable("Sorted matrix of observations:", res.get(ResultId.sortedData));

        context.dispose();
    }
}
