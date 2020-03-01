/* file: PCATransformDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 //     Java example of PCA transformation algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.pca_transform;



import com.intel.daal.algorithms.pca.Batch;
import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.algorithms.pca.ResultsToComputeId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import com.intel.daal.algorithms.pca.*;
import com.intel.daal.algorithms.pca.transform.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.pca.ResultId;

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCATRANSFORMDENSEBATCH">
 * @example PCATransformDenseBatch.java
 */

class PCATransformDenseBatch {
    private static final String dataset = "../data/batch/pca_transform.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException, java.lang.IllegalArgumentException {
        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                                                       DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                       DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();
        NumericTable input = dataSource.getNumericTable();

        /* Create a PCA algorithm */
        Batch algorithm =
            new Batch(context, Float.class, Method.correlationDense);
        /* Set an input object for the algorithm */
        algorithm.input.set(InputId.data, input);

        algorithm.parameter.setResultsToCompute(ResultsToComputeId.mean | ResultsToComputeId.variance | ResultsToComputeId.eigenvalue);

        /* Compute PCA */
        Result result = algorithm.compute();

        Service.printNumericTable("Eigenvalues:",result.get(ResultId.eigenValues));
        Service.printNumericTable("Eigenvectors:",result.get(ResultId.eigenVectors));
        Service.printNumericTable("Eigenvalues kv:",result.get(ResultId.eigenValues));
        Service.printNumericTable("Means kv:",result.get(ResultId.means));
        Service.printNumericTable("Variances kv:",result.get(ResultId.variances));

        KeyValueDataCollection dataCollection = new KeyValueDataCollection(context);

        /* Create a PCA transform algorithm */
        TransformBatch transformAlgorithm = new TransformBatch(context, Float.class, TransformMethod.defaultDense, 2);

        /* Set an input object for the algorithm */
        transformAlgorithm.input.set(TransformInputId.data, input);

        /* Set eigenvectors for the algorithm */
        transformAlgorithm.input.set(TransformInputId.eigenvectors, result.get(ResultId.eigenVectors));

        transformAlgorithm.input.set(TransformDataInputId.dataForTransform,dataCollection);

        /* Compute PCA transfromation */
        TransformResult transformResult = transformAlgorithm.compute();

        /* Print the results of stage */
        Service.printNumericTable("Transformed data:", transformResult.get(TransformResultId.transformedData));

        context.dispose();
    }
}
