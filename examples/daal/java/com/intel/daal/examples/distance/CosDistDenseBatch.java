/* file: CosDistDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 //     Java example of computing a cosine distance matrix
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-COSINEDISTANCEBATCH">
 * @example CosDistDenseBatch.java
 */

package com.intel.daal.examples.distance;

import com.intel.daal.algorithms.cosdistance.Batch;
import com.intel.daal.algorithms.cosdistance.InputId;
import com.intel.daal.algorithms.cosdistance.Method;
import com.intel.daal.algorithms.cosdistance.Result;
import com.intel.daal.algorithms.cosdistance.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class CosDistDenseBatch {
    /* Input data set parameters */
    private static final String dataset       = "../data/batch/distance.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm to compute a cosine distance matrix using the defaultDense method */
        Batch alg = new Batch(context, Float.class, Method.defaultDense);

        NumericTable input = dataSource.getNumericTable();
        alg.input.set(InputId.data, input);
        Result result = alg.compute();

        NumericTable res = result.get(ResultId.cosineDistance);

        Service.printNumericTable("Cosine distance", res, 15);

        context.dispose();
    }
}
