/* file: PCATransformDenseBatch.java */
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

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
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

        //KeyValueDataCollection resultCollection = result.get(TransformDataInputId.dataForTransform);

        Service.printNumericTable("Eigenvalues kv:",result.get(ResultId.eigenValues));
        Service.printNumericTable("Means kv:",result.get(ResultId.means));
        Service.printNumericTable("Variances kv:",result.get(ResultId.variances));

        long a=2;
        TransformInput inputDataAlg =  new TransformInput(context,a);
        //inputDataAlg.set(TransformInputId.eigenvectors, result.get(ResultId.eigenVectors));
        long b=3;
        KeyValueDataCollection dataCollection = new KeyValueDataCollection(context);

        /* Create a PCA transform algorithm */
        TransformBatch transformAlgorithm = new TransformBatch(context, Float.class, TransformMethod.defaultDense, 2);


        /* Set an input object for the algorithm */
        transformAlgorithm.input.set(TransformInputId.data, input);

        ResultId transformResultId = new ResultId(TransformDataInputId.dataForTransform.getValue());
        int id = transformResultId.getValue();
        //System.out.println(id);
        /* Set eigenvectors for the algorithm */
        transformAlgorithm.input.set(TransformInputId.eigenvectors, result.get(ResultId.eigenVectors));
        NumericTable trNumTable = result.get(ResultId.means);

        transformAlgorithm.input.set(TransformDataInputId.dataForTransform,dataCollection);

        /* Compute PCA transfromation */
        TransformResult transformResult = transformAlgorithm.compute();

        /* Print the results of stage */
        //Service.printNumericTable("First 4 rows of the input data:", input, 4);
        //Service.printNumericTable("First 4 rows of the PCA transformation result:",
        //     transformResult.get(TransformResultId.transformedData), 4);


       Service.printNumericTable("Transformed data:", transformResult.get(TransformResultId.transformedData));

        context.dispose();
    }
}
