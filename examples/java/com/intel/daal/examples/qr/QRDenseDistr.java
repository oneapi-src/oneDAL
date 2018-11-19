/* file: QRDenseDistr.java */
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
 //     Java example of computing QR decomposition in the distributed processing
 //     mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-QRDISTRIBUTED">
 * @example QRDenseDistr.java
 */

package com.intel.daal.examples.qr;

import com.intel.daal.algorithms.qr.*;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class QRDenseDistr {
    /* Input data set parameters */
    private static final String[] dataset         = { "../data/distributed/qr_1.csv", "../data/distributed/qr_2.csv",
            "../data/distributed/qr_3.csv", "../data/distributed/qr_4.csv" };
    private static final int      nNodes          = dataset.length;

    private static DataCollection[] dataFromStep1ForStep2 = new DataCollection[nNodes];
    private static DataCollection[] dataFromStep1ForStep3 = new DataCollection[nNodes];
    private static DataCollection[] dataFromStep2ForStep3 = new DataCollection[nNodes];

    private static KeyValueDataCollection inputForStep3FromStep2;

    private static NumericTable   R;
    private static NumericTable[] Qi = new NumericTable[nNodes];

    private static DistributedStep1Local  qrStep1Local;
    private static DistributedStep2Master qrStep2Master;
    private static DistributedStep3Local  qrStep3Local;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        for (int iNode = 0; iNode < nNodes; iNode++) {
            computeStep1Local(iNode);
        }

        computeStep2Master();

        for (int iNode = 0; iNode < nNodes; iNode++) {
            computeStep3Local(iNode);
        }

        /* Print the results */
        printResults();

        context.dispose();
    }

    static void computeStep1Local(int block) {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset[block],
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();

        NumericTable input = dataSource.getNumericTable();

        /* Create an algorithm to compute QR decomposition on local nodes */
        qrStep1Local = new DistributedStep1Local(context, Float.class, Method.defaultDense);
        qrStep1Local.input.set(InputId.data, input);

        /* Compute QR decomposition */
        DistributedStep1LocalPartialResult pres = qrStep1Local.compute();

        dataFromStep1ForStep2[block] = pres.get(PartialResultId.outputOfStep1ForStep2);
        dataFromStep1ForStep3[block] = pres.get(PartialResultId.outputOfStep1ForStep3);
    }

    static void computeStep2Master() {

        /* Create an algorithm to compute QR decomposition on the master node */
        qrStep2Master = new DistributedStep2Master(context, Float.class, Method.defaultDense);
        for (int iNode = 0; iNode < nNodes; iNode++) {
            qrStep2Master.input.add(DistributedStep2MasterInputId.inputOfStep2FromStep1, iNode,
                    dataFromStep1ForStep2[iNode]);
        }

        /* Compute QR decomposition */
        DistributedStep2MasterPartialResult pres = qrStep2Master.compute();

        inputForStep3FromStep2 = pres.get(DistributedPartialResultCollectionId.outputOfStep2ForStep3);

        for (int iNode = 0; iNode < nNodes; iNode++) {
            dataFromStep2ForStep3[iNode] = (DataCollection)inputForStep3FromStep2.get(iNode);
        }

        Result result = qrStep2Master.finalizeCompute();
        R = result.get(ResultId.matrixR);
    }

    private static void computeStep3Local(int block) {

        /* Create an algorithm to compute QR decomposition on the master node */
        qrStep3Local = new DistributedStep3Local(context, Float.class, Method.defaultDense);
        qrStep3Local.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep1, dataFromStep1ForStep3[block]);
        qrStep3Local.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep2, dataFromStep2ForStep3[block]);

        /* Compute QR decomposition */
        qrStep3Local.compute();

        Result result = qrStep3Local.finalizeCompute();

        Qi[block] = result.get(ResultId.matrixQ);
    }

    private static void printResults() {
        Service.printNumericTable("Part of orthogonal matrix Q from 1st node:", Qi[0], 10);
        Service.printNumericTable("Triangular matrix R:", R);
    }
}
