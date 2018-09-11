/* file: CovCSRDistr.java */
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
 //     Java example of variance-covariance matrix computation in the distributed
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-COVARIANCECSRDISTRIBUTED">
 * @example CovCSRDistr.java
 */

package com.intel.daal.examples.covariance;

import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/*
// Input data set is stored in the compressed sparse row format
*/

class CovCSRDistr {

    /* Input data set parameters */
    private static final String datasetFileNames[] = new String[] { "../data/distributed/covcormoments_csr_1.csv",
            "../data/distributed/covcormoments_csr_2.csv", "../data/distributed/covcormoments_csr_3.csv",
            "../data/distributed/covcormoments_csr_4.csv" };

    private static final int nBlocks = 4;

    private static PartialResult[] partialResult = new PartialResult[nBlocks];
    private static Result          result;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        for (int i = 0; i < nBlocks; i++) {
            computeOnLocalNode(i);
        }

        computeOnMasterNode();

        HomogenNumericTable covariance = (HomogenNumericTable) result.get(ResultId.covariance);
        HomogenNumericTable mean = (HomogenNumericTable) result.get(ResultId.mean);

        Service.printNumericTable("Covariance matrix (upper left square 10*10) :", covariance, 10, 10);
        Service.printNumericTable("Mean vector:", mean, 1, 10);

        context.dispose();
    }

    private static void computeOnLocalNode(int block) throws java.io.IOException {
        /* Read the input data from a file */
        CSRNumericTable dataTable = Service.createSparseTable(context, datasetFileNames[block]);

        /* Create algorithm objects to compute a variance-covariance matrix in the distributed processing mode using the default method */
        DistributedStep1Local algorithm = new DistributedStep1Local(context, Float.class, Method.fastCSR);

        /* Set input objects for the algorithm */
        algorithm.input.set(InputId.data, dataTable);

        /* Compute partial estimates on nodes */
        partialResult[block] = algorithm.compute();
    }

    private static void computeOnMasterNode() {
        /* Create algorithm objects to compute a variance-covariance matrix in the distributed processing mode using the default method */
        DistributedStep2Master algorithm = new DistributedStep2Master(context, Float.class, Method.fastCSR);

        /* Set input objects for the algorithm */
        for (int i = 0; i < nBlocks; i++) {
            algorithm.input.add(DistributedStep2MasterInputId.partialResults, partialResult[i]);
        }

        /* Compute a partial estimate on the master node from the partial estimates on local nodes */
        algorithm.compute();

        /* Finalize the result in the distributed processing mode */
        result = algorithm.finalizeCompute();
    }
}
