/* file: SVDDenseDistr.java */
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

        Service.printNumericTable("Singular values:", S);
        Service.printNumericTable("Right orthogonal matrix V:", V);
        Service.printNumericTable("Part of left orthogonal matrix U from 1st node:", U[0], 10);


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
        svdStep1Local = new DistributedStep1Local(context, Float.class, Method.defaultDense);

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
        svdStep2Master = new DistributedStep2Master(context, Float.class, Method.defaultDense);

        /* Set the results calculated in step 1 */
        for (int iNode = 0; iNode < nNodes; iNode++) {
            svdStep2Master.input.add(DistributedStep2MasterInputId.inputOfStep2FromStep1, iNode,
                    dataFromStep1ForStep2[iNode]);
        }

        /* Compute SVD */
        DistributedStep2MasterPartialResult pres = svdStep2Master.compute();

        /* Get the result for step 3 */
        for (int iNode = 0; iNode < nNodes; iNode++) {
            dataFromStep2ForStep3[iNode] = pres.get(DistributedPartialResultCollectionId.outputOfStep2ForStep3, iNode);
        }

        Result result = svdStep2Master.finalizeCompute();

        /* Get final singular values and a matrix of right singular vectors */
        S = result.get(ResultId.singularValues);
        V = result.get(ResultId.rightSingularMatrix);
    }

    static void computeStep3Local(int i) {
        /* Create an algorithm to compute SVD on local nodes */
        svdStep3Local = new DistributedStep3Local(context, Float.class, Method.defaultDense);

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
