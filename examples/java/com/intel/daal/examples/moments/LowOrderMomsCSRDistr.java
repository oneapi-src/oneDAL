/* file: LowOrderMomsCSRDistr.java */
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
 //     Java example of computing low order moments in the distributed processing
 //     mode.
 //
 //     Input matrix is stored in the compressed sparse row (CSR) format with
 //     one-based indexing.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOWORDERMOMENTSCSRDISTRIBUTED">
 * @example LowOrderMomsCSRDistr.java
 */

package com.intel.daal.examples.moments;

import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/*
// Input data set is stored in the compressed sparse row format
*/

class LowOrderMomsCSRDistr {

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

        printResults();

        context.dispose();
    }

    private static void computeOnLocalNode(int block) throws java.io.IOException {
        /* Read the input data from a file */
        CSRNumericTable dataTable = Service.createSparseTable(context, datasetFileNames[block]);

        /* Create algorithm objects to compute low order moments in the distributed processing mode using the default method */
        DistributedStep1Local algorithm = new DistributedStep1Local(context, Float.class, Method.fastCSR);

        /* Set input objects for the algorithm */
        algorithm.input.set(InputId.data, dataTable);

        /* Compute partial low order moments estimates on local nodes */
        partialResult[block] = algorithm.compute();
    }

    private static void computeOnMasterNode() {
        /* Create algorithm objects to compute low order moments in the distributed processing mode using the default method */
        DistributedStep2Master algorithm = new DistributedStep2Master(context, Float.class, Method.fastCSR);

        /* Set input objects for the algorithm */
        for (int i = 0; i < nBlocks; i++) {
            algorithm.input.add(DistributedStep2MasterInputId.partialResults, partialResult[i]);
        }

        /* Compute a partial low order moments estimate on the master node from the partial estimates on local nodes */
        algorithm.compute();

        /* Finalize the result in the distributed processing mode */
        result = algorithm.finalizeCompute();
    }

    private static void printResults() {
        HomogenNumericTable minimum = (HomogenNumericTable) result.get(ResultId.minimum);
        HomogenNumericTable maximum = (HomogenNumericTable) result.get(ResultId.maximum);
        HomogenNumericTable sum = (HomogenNumericTable) result.get(ResultId.sum);
        HomogenNumericTable sumSquares = (HomogenNumericTable) result.get(ResultId.sumSquares);
        HomogenNumericTable sumSquaresCentered = (HomogenNumericTable) result.get(ResultId.sumSquaresCentered);
        HomogenNumericTable mean = (HomogenNumericTable) result.get(ResultId.mean);
        HomogenNumericTable secondOrderRawMoment = (HomogenNumericTable) result.get(ResultId.secondOrderRawMoment);
        HomogenNumericTable variance = (HomogenNumericTable) result.get(ResultId.variance);
        HomogenNumericTable standardDeviation = (HomogenNumericTable) result.get(ResultId.standardDeviation);
        HomogenNumericTable variation = (HomogenNumericTable) result.get(ResultId.variation);

        Service.printNumericTable("Minimum:", minimum);
        Service.printNumericTable("Maximum:", maximum);
        Service.printNumericTable("Sum:", sum);
        Service.printNumericTable("Sum of squares:", sumSquares);
        Service.printNumericTable("Sum of squared difference from the means:", sumSquaresCentered);
        Service.printNumericTable("Mean:", mean);
        Service.printNumericTable("Second order raw moment:", secondOrderRawMoment);
        Service.printNumericTable("Variance:", variance);
        Service.printNumericTable("Standart deviation:", standardDeviation);
        Service.printNumericTable("Variation:", variation);
    }
}
