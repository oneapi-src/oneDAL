/* file: SortingDenseBatch.java */
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
        Batch algorithm = new Batch(context, Double.class, Method.defaultDense);
        algorithm.input.set(InputId.data, dataSource.getNumericTable());

        /* Sort data observations */
        Result res = algorithm.compute();

        Service.printNumericTable("Sorted matrix of observations:", res.get(ResultId.sortedData));

        context.dispose();
    }
}
