/* file: PCACorDenseDistr.java */
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
 //     Java example of principal component analysis (PCA) using the correlation
 //     method in the distributed processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCACORRELATIONDENSEDISTRIBUTED">
 * @example PCACorDenseDistr.java
 */

package com.intel.daal.examples.pca;

import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.pca.DistributedStep1Local;
import com.intel.daal.algorithms.pca.DistributedStep2Master;
import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.MasterInputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class PCACorDenseDistr {
    /* Input data set parameters */
    private static final String[] dataset = { "../data/distributed/pca_normalized_1.csv",
                                              "../data/distributed/pca_normalized_2.csv", "../data/distributed/pca_normalized_3.csv",
                                              "../data/distributed/pca_normalized_4.csv",
                                            };
    private static final int nNodes = 4;

    private static PartialResult[] pres = new PartialResult[nNodes];

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        for (int i = 0; i < nNodes; i++) {
            DaalContext localContext = new DaalContext();

            /* Initialize FileDataSource to retrieve the input data from a .csv file */
            FileDataSource dataSource = new FileDataSource(localContext, dataset[i],
                                                           DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                           DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

            /* Retrieve the data from the input file */
            dataSource.loadDataBlock();

            /* Create an algorithm to compute PCA decomposition using the correlation method on local nodes */
            DistributedStep1Local pcaLocal = new DistributedStep1Local(localContext, Double.class,
                                                                       Method.correlationDense);

            /* Set the input data on local nodes */
            NumericTable data = dataSource.getNumericTable();
            pcaLocal.input.set(InputId.data, data);

            /* Compute PCA decomposition on local nodes */
            pres[i] = pcaLocal.compute();
            pres[i].pack();

            localContext.dispose();
        }

        /* Create an algorithm to compute PCA decomposition using the correlation method on the master node */
        DistributedStep2Master pcaMaster = new DistributedStep2Master(context, Double.class, Method.correlationDense);

        /* Add partial results computed on local nodes to the algorithm on the master node */
        for (int i = 0; i < nNodes; i++) {
            pres[i].unpack(context);
            pcaMaster.input.add(MasterInputId.partialResults, pres[i]);
        }

        /* Compute PCA decomposition on the master node */
        pcaMaster.compute();

        /* Finalize computations and retrieve the results */
        Result res = pcaMaster.finalizeCompute();

        NumericTable eigenValues = res.get(ResultId.eigenValues);
        NumericTable eigenVectors = res.get(ResultId.eigenVectors);
        Service.printNumericTable("Eigenvalues:", eigenValues);
        Service.printNumericTable("Eigenvectors:", eigenVectors);

        context.dispose();
    }
}
