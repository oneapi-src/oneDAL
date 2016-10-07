/* file: SVDDenseDistr.java */
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
 //     Java example of singular value decomposition (SVD) in the distributed
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SVDDISTRIBUTED">
 * @example SVDDenseDistr.java
 */

package com.intel.daal.examples.svd;

import com.intel.daal.algorithms.svd.*;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class SVDDenseDistr {
    /* Input data set parameters */
    private static final String[] dataset = { "../data/distributed/svd_1.csv", "../data/distributed/svd_2.csv",
            "../data/distributed/svd_3.csv", "../data/distributed/svd_4.csv" };

    private static final int nNodes          = dataset.length;

    private static DataCollection[] dataFromStep1ForStep2 = new DataCollection[nNodes];
    private static DataCollection[] dataFromStep1ForStep3 = new DataCollection[nNodes];
    private static DataCollection[] dataFromStep2ForStep3 = new DataCollection[nNodes];

    private static KeyValueDataCollection inputForStep3FromStep2;

    private static NumericTable   S;
    private static NumericTable   V;
    private static NumericTable[] U = new NumericTable[nNodes];

    private static DistributedStep1Local  svdStep1Local;
    private static DistributedStep2Master svdStep2Master;
    private static DistributedStep3Local  svdStep3Local;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        for (int iNode = 0; iNode < nNodes; iNode++) {
            computeStep1Local(iNode);
        }

        computeStep2Master();

        for (int iNode = 0; iNode < nNodes; iNode++) {
            computeStep3Local(iNode);
        }

        Service.printNumericTable("Left orthogonal matrix U (10 first vectors):", U[0], 10);
        Service.printNumericTable("Singular values:", S);
        Service.printNumericTable("Right orthogonal matrix V:", V);

        context.dispose();
    }

    static void computeStep1Local(int i) {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset[i],
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the input data */
        dataSource.loadDataBlock();

        /* Create an algorithm to compute SVD on local nodes */
        svdStep1Local = new DistributedStep1Local(context, Double.class, Method.defaultDense);

        /* Set the input data on local nodes */
        NumericTable input = dataSource.getNumericTable();
        svdStep1Local.input.set(InputId.data, input);

        /* Compute SVD */
        DistributedStep1LocalPartialResult pres = svdStep1Local.compute();

        /* Get the results for next steps */
        dataFromStep1ForStep2[i] = pres.get(PartialResultId.outputOfStep1ForStep2);
        dataFromStep1ForStep3[i] = pres.get(PartialResultId.outputOfStep1ForStep3);
    }

    static void computeStep2Master() {
        /* Create an algorithm to compute SVD on the master node */
        svdStep2Master = new DistributedStep2Master(context, Double.class, Method.defaultDense);

        /* Set the results calculated in step 1 */
        for (int iNode = 0; iNode < nNodes; iNode++) {
            svdStep2Master.input.add(DistributedStep2MasterInputId.inputOfStep2FromStep1, iNode,
                    dataFromStep1ForStep2[iNode]);
        }

        /* Compute SVD */
        DistributedStep2MasterPartialResult pres = svdStep2Master.compute();

        /* Get the result for step 3 */
        inputForStep3FromStep2 = pres.get(DistributedPartialResultCollectionId.outputOfStep2ForStep3);

        for (int iNode = 0; iNode < nNodes; iNode++) {
            dataFromStep2ForStep3[iNode] = (DataCollection)inputForStep3FromStep2.get(iNode);
        }

        Result result = svdStep2Master.finalizeCompute();

        /* Get final singular values and a matrix of right singular vectors */
        S = result.get(ResultId.singularValues);
        V = result.get(ResultId.rightSingularMatrix);
    }

    static void computeStep3Local(int i) {
        /* Create an algorithm to compute SVD on local nodes */
        svdStep3Local = new DistributedStep3Local(context, Double.class, Method.defaultDense);

        /* Set the results calculated in steps 1 and 3 */
        svdStep3Local.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep1, dataFromStep1ForStep3[i]);
        svdStep3Local.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep2, dataFromStep2ForStep3[i]);

        /* Compute SVD */
        svdStep3Local.compute();
        Result result = svdStep3Local.finalizeCompute();

        /* Get final matrices of left singular vectors */
        U[i] = result.get(ResultId.leftSingularMatrix);
    }
}
