/* file: QuantilesDenseBatch.java */
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
