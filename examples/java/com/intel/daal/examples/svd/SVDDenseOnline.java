/* file: SVDDenseOnline.java */
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
 //     Java example of singular value decomposition (SVD) in the online
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SVDONLINE">
 * @example SVDDenseOnline.java
 */

package com.intel.daal.examples.svd;

import com.intel.daal.algorithms.svd.InputId;
import com.intel.daal.algorithms.svd.Method;
import com.intel.daal.algorithms.svd.Online;
import com.intel.daal.algorithms.svd.Result;
import com.intel.daal.algorithms.svd.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class SVDDenseOnline {
    /* Input data set parameters */
    private static final String dataset         = "../data/online/svd.csv";
    private static final int    nVectorsInBlock = 4000;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Create an algorithm to compute SVD in the online processing mode */
        Online svdAlgorithm = new Online(context, Double.class, Method.defaultDense);

        /* Set the input data to the SVD algorithm */
        NumericTable input = dataSource.getNumericTable();
        svdAlgorithm.input.set(InputId.data, input);

        while (dataSource.loadDataBlock(nVectorsInBlock) == nVectorsInBlock) {
            /* Compute SVD */
            svdAlgorithm.compute();
        }

        Result res = svdAlgorithm.finalizeCompute();

        /* Print the results */
        NumericTable leftSingularMatrix = res.get(ResultId.leftSingularMatrix);
        NumericTable singularValues = res.get(ResultId.singularValues);
        NumericTable rightSingularMatrix = res.get(ResultId.rightSingularMatrix);

        Service.printNumericTable("Left orthogonal matrix U (10 first vectors):", leftSingularMatrix, 10);
        Service.printNumericTable("Singular values:", singularValues);
        Service.printNumericTable("Right orthogonal matrix V:", rightSingularMatrix);

        context.dispose();
    }
}
