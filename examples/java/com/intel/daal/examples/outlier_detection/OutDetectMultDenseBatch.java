/* file: OutDetectMultDenseBatch.java */
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
 //     Java example of multivariate outlier detection
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-OUTLIERDETECTIONMULTIVARIATEDENSEBATCH">
 * @example OutDetectMultDenseBatch.java
 */

package com.intel.daal.examples.outlier_detection;

import com.intel.daal.algorithms.multivariate_outlier_detection.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;

import com.intel.daal.services.DaalContext;

class OutDetectMultDenseBatch {

    /* Input data set parameters */
    private static final String datasetFileName = "../data/batch/outlierdetection.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();

        /* Create an algorithm to detect outliers using the default method */
        Batch alg = new Batch(context, Double.class, Method.defaultDense);

        NumericTable data = dataSource.getNumericTable();
        alg.input.set(InputId.data, data);

        /* Detect outliers */
        Result result = alg.compute();

        NumericTable weights = result.get(ResultId.weights);

        Service.printNumericTables(data,weights,"Input data","Weights","Outlier detection result (Default method)",0);
        context.dispose();
    }
}
