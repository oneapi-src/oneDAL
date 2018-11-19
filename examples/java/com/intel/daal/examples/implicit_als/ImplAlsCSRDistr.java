/* ImplicitAlsCSRDistributed.java */
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
 //     Java example of the implicit alternating least squares (ALS) algorithm in
 //     the distributed processing mode.
 //
 //     The program trains the implicit ALS model on a training data set.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-IMPLICITALSCSRDISTRIBUTED">
 * @example ImplAlsCSRDistr.java
 */

package com.intel.daal.examples.implicit_als;

import com.intel.daal.algorithms.implicit_als.PartialModel;
import com.intel.daal.algorithms.implicit_als.prediction.ratings.*;
import com.intel.daal.algorithms.implicit_als.training.*;
import com.intel.daal.algorithms.implicit_als.training.init.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class ImplAlsCSRDistr {
    /* Input data set parameters */
    private static final int      nNodes            = 4;
    private static final String[] dataset = {
            "../data/distributed/implicit_als_trans_csr_1.csv",
            "../data/distributed/implicit_als_trans_csr_2.csv",
            "../data/distributed/implicit_als_trans_csr_3.csv",
            "../data/distributed/implicit_als_trans_csr_4.csv" };

    private static final int    nUsers           = 46;           /* Full number of users */
    private static final int    nFactors         = 2;            /* Number of factors */
    private static final int    maxIterations    = 5;            /* Number of iterations in the implicit ALS training algorithm */

    private static NumericTable[] dataTable           = new NumericTable[nNodes];
    private static NumericTable[] dataTableTransposed = new NumericTable[nNodes];

    private static long[] usersPartition = { nNodes };

    private static NumericTable[] userOffsets = new NumericTable[nNodes];
    private static NumericTable[] itemOffsets = new NumericTable[nNodes];

    private static DistributedPartialResultStep1[] step1LocalResult = new DistributedPartialResultStep1[nNodes];
    private static NumericTable step2MasterResult;

    private static KeyValueDataCollection[] userStep3LocalInput = new KeyValueDataCollection[nNodes];
    private static KeyValueDataCollection[] itemStep3LocalInput = new KeyValueDataCollection[nNodes];

    private static KeyValueDataCollection[] step3LocalResult = new KeyValueDataCollection[nNodes];
    private static KeyValueDataCollection[] step4LocalInput  = new KeyValueDataCollection[nNodes];

    private static DistributedPartialResultStep4[] itemsPartialResultLocal = new DistributedPartialResultStep4[nNodes];
    private static DistributedPartialResultStep4[] usersPartialResultLocal = new DistributedPartialResultStep4[nNodes];

    private static NumericTable[][] predictedRatings = new NumericTable[nNodes][];

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        for (int iNode = 0; iNode < nNodes; iNode++) {
            readData(iNode);
            step4LocalInput[iNode] = new KeyValueDataCollection(context);
        }

        initializeModel();

        trainModel();

        testModel();

        printResults();

        context.dispose();
    }

    static KeyValueDataCollection initializeStep1Local(int iNode) {

        /* Create an algorithm object to perform the first step of the implicit ALS initialization algorithm */
        InitDistributedStep1Local initAlgorithm = new InitDistributedStep1Local(context, Float.class, InitMethod.fastCSR);
        initAlgorithm.parameter.setFullNUsers(nUsers);
        initAlgorithm.parameter.setNFactors(nFactors);
        initAlgorithm.parameter.setSeed(initAlgorithm.parameter.getSeed() + iNode);
        initAlgorithm.parameter.setPartition(new HomogenNumericTable(context, usersPartition, 1, usersPartition.length));

        /* Pass a training data set to the algorithm */
        initAlgorithm.input.set(InitInputId.data, dataTable[iNode]);

        /* Compute partial results of the first step on local nodes */
        InitPartialResult initPartialResult = initAlgorithm.compute();

        itemStep3LocalInput[iNode] = initPartialResult.get(InitPartialResultBaseId.outputOfInitForComputeStep3);
        userOffsets[iNode]         = initPartialResult.get(InitPartialResultBaseId.offsets, iNode);
        PartialModel partialModel  = initPartialResult.get(InitPartialResultId.partialModel);

        itemsPartialResultLocal[iNode] = new DistributedPartialResultStep4(context);
        itemsPartialResultLocal[iNode].set(DistributedPartialResultStep4Id.outputOfStep4ForStep1, partialModel);

        return initPartialResult.get(InitPartialResultCollectionId.outputOfStep1ForStep2);
    }

    static void initializeStep2Local(int iNode, KeyValueDataCollection initStep2LocalInput) {

        /* Create an algorithm object to perform the second step of the implicit ALS initialization algorithm */
        InitDistributedStep2Local initAlgorithm = new InitDistributedStep2Local(context, Float.class, InitMethod.fastCSR);

        initAlgorithm.input.set(InitStep2LocalInputId.inputOfStep2FromStep1, initStep2LocalInput);

        /* Compute partial results of the second step on local nodes */
        InitDistributedPartialResultStep2 initPartialResult = initAlgorithm.compute();

        dataTableTransposed[iNode] = initPartialResult.get(InitDistributedPartialResultStep2Id.transposedData);
        userStep3LocalInput[iNode] = initPartialResult.get(InitPartialResultBaseId.outputOfInitForComputeStep3);
        itemOffsets[iNode]         = initPartialResult.get(InitPartialResultBaseId.offsets, iNode);
    }

    static void initializeModel()
    {
        KeyValueDataCollection[] initStep1LocalResult = new KeyValueDataCollection[nNodes];
        for (int iNode = 0; iNode < nNodes; iNode++) {
            initStep1LocalResult[iNode] = initializeStep1Local(iNode);
        }

        /* Prepare input objects for the second step of the distributed initialization algorithm */
        KeyValueDataCollection[] initStep2LocalInput = new KeyValueDataCollection[nNodes];
        for (int iNode = 0; iNode < nNodes; iNode++) {
            initStep2LocalInput[iNode] = new KeyValueDataCollection(context);
            for (int jNode = 0; jNode < nNodes; jNode++) {
                initStep2LocalInput[iNode].set(jNode, initStep1LocalResult[jNode].get(iNode));
            }
        }

        for (int iNode = 0; iNode < nNodes; iNode++) {
            initializeStep2Local(iNode, initStep2LocalInput[iNode]);
        }
    }

    static void trainModel() {

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            /* Update partial users factors */
            for (int iNode = 0; iNode < nNodes; iNode++) {
                computeStep1Local(iNode, itemsPartialResultLocal);
            }

            computeStep2Master();

            for (int iNode = 0; iNode < nNodes; iNode++) {
                computeStep3Local(iNode, itemOffsets[iNode], itemsPartialResultLocal, itemStep3LocalInput);
            }

            for (int iNode = 0; iNode < nNodes; iNode++) {
                computeStep4Local(iNode, dataTableTransposed, usersPartialResultLocal);
            }

            /* Update partial items factors */
            for (int iNode = 0; iNode < nNodes; iNode++) {
                computeStep1Local(iNode, usersPartialResultLocal);
            }

            computeStep2Master();

            for (int iNode = 0; iNode < nNodes; iNode++) {
                computeStep3Local(iNode, userOffsets[iNode], usersPartialResultLocal, userStep3LocalInput);
            }

            for (int iNode = 0; iNode < nNodes; iNode++) {
                computeStep4Local(iNode, dataTable, itemsPartialResultLocal);
            }
        }
    }

    static void computeStep1Local(int iNode, DistributedPartialResultStep4[] partialResultLocal) {
        /* Create an algorithm object to perform first step of the implicit ALS training algorithm on local-node data */
        DistributedStep1Local algorithm = new DistributedStep1Local(context, Float.class, TrainingMethod.fastCSR);
        algorithm.parameter.setNFactors(nFactors);

        /* Set input objects for the algorithm */
        algorithm.input.set(PartialModelInputId.partialModel,
                            partialResultLocal[iNode].get(DistributedPartialResultStep4Id.outputOfStep4ForStep1));

        /* Compute partial results of the first step on local nodes */
        step1LocalResult[iNode] = algorithm.compute();
    }

    static void computeStep2Master() {
        /* Create an algorithm object to perform second step of the implicit ALS training algorithm */
        DistributedStep2Master algorithm = new DistributedStep2Master(context, Float.class, TrainingMethod.fastCSR);
        algorithm.parameter.setNFactors(nFactors);

        /* Set the partial results of the first local step of distributed computations
           as input for the master-node algorithm */
        for (int i = 0; i < nNodes; i++)
        {
            algorithm.input.add(MasterInputId.inputOfStep2FromStep1, step1LocalResult[i]);
        }

        /* Compute a partial result on the master node from the partial results on local nodes */
        step2MasterResult = algorithm.compute().get(DistributedPartialResultStep2Id.outputOfStep2ForStep4);
    }

    static void computeStep3Local(int iNode, NumericTable offset, DistributedPartialResultStep4[] partialResultLocal,
                                  KeyValueDataCollection[] outBlocks) {
        /* Create an algorithm object to perform third step of the implicit ALS training algorithm on local-node data */
        DistributedStep3Local algorithm = new DistributedStep3Local(context, Float.class, TrainingMethod.fastCSR);
        algorithm.parameter.setNFactors(nFactors);

        /* Set input objects for the algorithm */
        algorithm.input.set(PartialModelInputId.partialModel,
                            partialResultLocal[iNode].get(DistributedPartialResultStep4Id.outputOfStep4ForStep3));
        algorithm.input.set(Step3LocalCollectionInputId.partialModelBlocksToNode, outBlocks[iNode]);
        algorithm.input.set(Step3LocalNumericTableInputId.offset, offset);

        /* Compute partial results of the third step on local nodes */
        DistributedPartialResultStep3 partialResult = algorithm.compute();

        /* Prepare input objects for the fourth step of the distributed algorithm */
        step3LocalResult[iNode] = partialResult.get(DistributedPartialResultStep3Id.outputOfStep3ForStep4);
        for (int i = 0; i < nNodes; i++) {
            step4LocalInput[i].set(iNode, step3LocalResult[iNode].get(i));
        }
    }

    static void computeStep4Local(int iNode, NumericTable[] dataTable, DistributedPartialResultStep4[] partialResultLocal) {
        /* Create an algorithm object to perform fourth step of the implicit ALS training algorithm on local-node data */
        DistributedStep4Local algorithm = new DistributedStep4Local(context, Float.class, TrainingMethod.fastCSR);
        algorithm.parameter.setNFactors(nFactors);

        /* Set input objects for the algorithm */
        algorithm.input.set(Step4LocalPartialModelsInputId.partialModels,        step4LocalInput[iNode]);
        algorithm.input.set(Step4LocalNumericTableInputId.partialData,           dataTable[iNode]);
        algorithm.input.set(Step4LocalNumericTableInputId.inputOfStep4FromStep2, step2MasterResult);

        /* Get the local implicit ALS partial models */
        partialResultLocal[iNode] = algorithm.compute();
    }

    private static void testModel() {

        for (int iNode = 0; iNode < nNodes; iNode++) {
            predictedRatings[iNode] = new NumericTable[nNodes];
            for (int jNode = 0; jNode < nNodes; jNode++) {
                /* Create an algorithm object to predict ratings based in the implicit ALS partial models */
                RatingsDistributed algorithm = new RatingsDistributed(context, Float.class, RatingsMethod.defaultDense);
                algorithm.parameter.setNFactors(nFactors);

                /* Set input objects for the algorithm */
                algorithm.input.set(RatingsPartialModelInputId.usersPartialModel,
                        usersPartialResultLocal[iNode].get(DistributedPartialResultStep4Id.outputOfStep4));
                algorithm.input.set(RatingsPartialModelInputId.itemsPartialModel,
                        itemsPartialResultLocal[jNode].get(DistributedPartialResultStep4Id.outputOfStep4));

                /* Predict ratings */
                algorithm.compute();

                /* Retrieve the algorithm results */
                predictedRatings[iNode][jNode] = algorithm.finalizeCompute().get(RatingsResultId.prediction);
            }
        }
    }

    private static void readData(int iNode) throws java.io.FileNotFoundException, java.io.IOException {

        /* Read dataset from a file and create a numeric table for storing the input data */
        dataTable[iNode] = Service.createSparseTable(context, dataset[iNode]);
    }

    private static void printResults() {
        for (int iNode = 0; iNode < nNodes; iNode++) {
            for (int jNode = 0; jNode < nNodes; jNode++) {
                System.out.println("Ratings for users block " + iNode + ", items block " + jNode + " :");
                Service.printALSRatings(userOffsets[iNode], itemOffsets[jNode], predictedRatings[iNode][jNode]);
            }
        }
    }
}
