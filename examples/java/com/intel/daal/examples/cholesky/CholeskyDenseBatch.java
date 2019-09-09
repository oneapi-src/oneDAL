/* file: CholeskyDenseBatch.java */
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
 //     Java example of Cholesky decomposition
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.cholesky;

import com.intel.daal.algorithms.cholesky.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-CHOLESKYBATCH">
 * @example CholeskyDenseBatch.java
 */

class CholeskyDenseBatch {
    private static final String dataset   = "../data/batch/cholesky.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm */
        Batch choleskyAlgorithm = new Batch(context, Float.class, Method.defaultDense);

        /* Set an input object for the algorithm */
        NumericTable input = dataSource.getNumericTable();
        choleskyAlgorithm.input.set(InputId.data, input);

        /* Compute Cholesky decomposition */
        Result result = choleskyAlgorithm.compute();

        NumericTable choleskyFactor = result.get(ResultId.choleskyFactor);

        Service.printNumericTable("", choleskyFactor);

        context.dispose();
    }
}
