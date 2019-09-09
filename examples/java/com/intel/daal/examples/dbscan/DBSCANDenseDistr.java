/* file: DBSCANDenseDistr.java */
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
 //     Java example of dense DBSCAN clustering in the distributed processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DBSCANDENSEDISTR">
 * @example DBSCANDenseDistr.java
 */

package com.intel.daal.examples.dbscan;

import java.util.Vector;

import com.intel.daal.algorithms.dbscan.*;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class DBSCANDenseDistr {
    /* Input data set parameters */
    private static final int nBlocks = 4;
    private static final String[] dataset = {
            "../data/distributed/dbscan_dense_1.csv",
            "../data/distributed/dbscan_dense_2.csv",
            "../data/distributed/dbscan_dense_3.csv",
            "../data/distributed/dbscan_dense_4.csv"};

    private static final double epsilon = 0.02;
    private static final long   minObservations = 180;

    private static NumericTable[] dataTable = new NumericTable[nBlocks];

    private static DataCollection[] partitionedData = new DataCollection[nBlocks];
    private static DataCollection[] partitionedPartialOrders = new DataCollection[nBlocks];

    private static DataCollection[] partialSplits = new DataCollection[nBlocks];
    private static DataCollection[] partialBoundingBoxes = new DataCollection[nBlocks];
    private static DataCollection[] newPartitionedData = new DataCollection[nBlocks];
    private static DataCollection[] newPartitionedDataIndices = new DataCollection[nBlocks];
    private static DataCollection[] newPartitionedPartialOrders = new DataCollection[nBlocks];

    private static DataCollection[] haloData = new DataCollection[nBlocks];
    private static DataCollection[] haloDataIndices = new DataCollection[nBlocks];
    private static DataCollection[] haloBlocks = new DataCollection[nBlocks];

    private static DataCollection[] queries = new DataCollection[nBlocks];
    private static DataCollection[] newQueries = new DataCollection[nBlocks];

    private static DataCollection[] assignmentQueries = new DataCollection[nBlocks];
    private static DataCollection[] newAssignmentQueries = new DataCollection[nBlocks];

    private static NumericTable[] clusterStructure = new NumericTable[nBlocks];
    private static NumericTable[] finishedFlag = new NumericTable[nBlocks];
    private static NumericTable[] nClusters = new NumericTable[nBlocks];
    private static NumericTable[] clusterOffset = new NumericTable[nBlocks];
    private static NumericTable[] assignments = new NumericTable[nBlocks];
    private static NumericTable totalNClusters;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        for (int iNode = 0; iNode < nBlocks; iNode++) {
            readData(iNode);
        }

        geometricPartitioning();

        clustering();

        printResults();

        context.dispose();
    }

    private static void geometricPartitioning() {
        for (int block = 0; block < nBlocks; block++) {
            DistributedStep1Local step1 = new DistributedStep1Local(context, Float.class, Method.defaultDense, block, nBlocks);
            step1.input.set(Step1LocalNumericTableInputId.step1Data, dataTable[block]);
            DistributedPartialResultStep1 partialResult = step1.compute();

            partitionedData[block] = new DataCollection(context);
            partitionedPartialOrders[block] = new DataCollection(context);

            partitionedData[block].pushBack(dataTable[block]);
            partitionedPartialOrders[block].pushBack(partialResult.get(DistributedPartialResultStep1Id.partialOrder));
        }

        Vector<Integer> comsBegin = new Vector<>();
        Vector<Integer> comsEnd = new Vector<>();
        comsBegin.add(0);
        comsEnd.add(nBlocks);

        while (!comsBegin.isEmpty ()) {
            Vector<Integer> newComsBegin = new Vector<>();
            Vector<Integer> newComsEnd = new Vector<>();

            for (int comId = 0; comId < comsBegin.size(); comId++) {
                int beginBlock = comsBegin.get(comId);
                int endBlock = comsEnd.get(comId);
                int curNBlocks = endBlock - beginBlock;

                if (curNBlocks == 1) { continue; }

                for (int block = 0; block < curNBlocks; block++) {
                    partialSplits[block + beginBlock] = new DataCollection(context);
                    partialBoundingBoxes[block + beginBlock] = new DataCollection(context);
                    newPartitionedData[block + beginBlock] = new DataCollection(context);
                    newPartitionedPartialOrders[block + beginBlock] = new DataCollection(context);
                }

                for (int block = 0; block < curNBlocks; block++) {
                    DistributedStep2Local step2 = new DistributedStep2Local(context, Float.class, Method.defaultDense, block, nBlocks);
                    step2.input.set(LocalCollectionInputId.partialData, partitionedData[block + beginBlock]);
                    DistributedPartialResultStep2 partialResult = step2.compute();
                    NumericTable curBoundingBox = partialResult.get(DistributedPartialResultStep2Id.boundingBox);

                    for (int destBlock = 0; destBlock < curNBlocks; destBlock++) {
                        partialBoundingBoxes[destBlock + beginBlock].pushBack(curBoundingBox);
                    }
                }

                int leftBlocks = curNBlocks / 2;
                int rightBlocks = curNBlocks - leftBlocks;

                for (int block = 0; block < curNBlocks; block++) {
                    DistributedStep3Local step3 = new DistributedStep3Local(context, Float.class, Method.defaultDense, leftBlocks, rightBlocks);
                    step3.input.set(LocalCollectionInputId.partialData, partitionedData[block + beginBlock]);
                    step3.input.set(Step3LocalCollectionInputId.step3PartialBoundingBoxes, partialBoundingBoxes[block + beginBlock]);
                    DistributedPartialResultStep3 partialResult = step3.compute();
                    NumericTable curSplit = partialResult.get(DistributedPartialResultStep3Id.split);

                    for (int destBlock = 0; destBlock < curNBlocks; destBlock++) {
                        partialSplits[destBlock + beginBlock].pushBack(curSplit);
                    }
                }

                for (int block = 0; block < curNBlocks; block++) {
                    DistributedStep4Local step4 = new DistributedStep4Local(context, Float.class, Method.defaultDense, leftBlocks, rightBlocks);
                    step4.input.set(LocalCollectionInputId.partialData, partitionedData[block + beginBlock]);
                    step4.input.set(Step4LocalCollectionInputId.step4PartialOrders, partitionedPartialOrders[block + beginBlock]);
                    step4.input.set(Step4LocalCollectionInputId.step4PartialSplits, partialSplits[block + beginBlock]);
                    DistributedPartialResultStep4 partialResult = step4.compute();

                    DataCollection curPartitionedData = partialResult.get(DistributedPartialResultStep4Id.partitionedData);
                    DataCollection curPartitionedPartialOrders = partialResult.get(DistributedPartialResultStep4Id.partitionedPartialOrders);

                    for (int destBlock = 0; destBlock < curNBlocks; destBlock++) {
                        newPartitionedData[destBlock + beginBlock].pushBack(curPartitionedData.get(destBlock));
                        newPartitionedPartialOrders[destBlock + beginBlock].pushBack(curPartitionedPartialOrders.get(destBlock));
                    }
                }

                for (int block = 0; block < curNBlocks; block++) {
                    partitionedData[block + beginBlock] = newPartitionedData[block + beginBlock];
                    partitionedPartialOrders[block + beginBlock] = newPartitionedPartialOrders[block + beginBlock];
                }

                newComsBegin.add(beginBlock);
                newComsBegin.add(beginBlock + leftBlocks);
                newComsEnd.add(beginBlock + leftBlocks);
                newComsEnd.add(endBlock);
            }

            comsBegin = newComsBegin;
            comsEnd = newComsEnd;
        }
    }

    private static void clustering() {
        for (int block = 0; block < nBlocks; block++) {
            partialBoundingBoxes[block] = new DataCollection(context);
            haloData[block] = new DataCollection(context);
            haloDataIndices[block] = new DataCollection(context);
            haloBlocks[block] = new DataCollection(context);
        }

        for (int block = 0; block < nBlocks; block++) {
            DistributedStep2Local step2 = new DistributedStep2Local(context, Float.class, Method.defaultDense, block, nBlocks);
            step2.input.set(LocalCollectionInputId.partialData, partitionedData[block]);
            DistributedPartialResultStep2 partialResult = step2.compute();
            NumericTable curBoundingBox = partialResult.get(DistributedPartialResultStep2Id.boundingBox);

            for (int destBlock = 0; destBlock < nBlocks; destBlock++) {
                partialBoundingBoxes[destBlock].pushBack(curBoundingBox);
            }
        }

        for (int block = 0; block < nBlocks; block++) {
            DistributedStep5Local step5 = new DistributedStep5Local(context, Float.class, Method.defaultDense, block, nBlocks, epsilon);
            step5.input.set(LocalCollectionInputId.partialData, partitionedData[block]);
            step5.input.set(Step5LocalCollectionInputId.step5PartialBoundingBoxes, partialBoundingBoxes[block]);
            DistributedPartialResultStep5 partialResult = step5.compute();
            DataCollection curHaloData = partialResult.get(DistributedPartialResultStep5Id.partitionedHaloData);
            DataCollection curHaloDataIndices = partialResult.get(DistributedPartialResultStep5Id.partitionedHaloDataIndices);

            for (int destBlock = 0; destBlock < nBlocks; destBlock++) {
                NumericTable dataTable = (NumericTable)curHaloData.get(destBlock);
                NumericTable dataIndicesTable = (NumericTable)curHaloDataIndices.get(destBlock);
                if (dataTable.getNumberOfRows() > 0) {
                    haloData[destBlock].pushBack(dataTable);
                    haloDataIndices[destBlock].pushBack(dataIndicesTable);
                    haloBlocks[destBlock].pushBack(new HomogenNumericTable(context, Integer.class, 1, 1, NumericTable.AllocationFlag.DoAllocate, (int)block));
                }
            }
        }

        for (int block = 0; block < nBlocks; block++)
        {
            queries[block] = new DataCollection(context);
        }

        for (int block = 0; block < nBlocks; block++)
        {
            DistributedStep6Local step6 = new DistributedStep6Local(context, Float.class, Method.defaultDense, block, nBlocks, epsilon, minObservations);

            step6.input.set(LocalCollectionInputId.partialData, partitionedData[block]);
            step6.input.set(Step6LocalCollectionInputId.haloData, haloData[block]);
            step6.input.set(Step6LocalCollectionInputId.haloDataIndices, haloDataIndices[block]);
            step6.input.set(Step6LocalCollectionInputId.haloBlocks, haloBlocks[block]);
            DistributedPartialResultStep6 partialResult = step6.compute();

            clusterStructure[block] = partialResult.get(DistributedPartialResultStep6NumericTableId.step6ClusterStructure);
            finishedFlag[block] = partialResult.get(DistributedPartialResultStep6NumericTableId.step6FinishedFlag);
            nClusters[block] = partialResult.get(DistributedPartialResultStep6NumericTableId.step6NClusters);

            DataCollection curQueries = partialResult.get(DistributedPartialResultStep6CollectionId.step6Queries);

            for (int destBlock = 0; destBlock < nBlocks; destBlock++) {
                NumericTable table = (NumericTable)curQueries.get(destBlock);
                if (table.getNumberOfRows() > 0) {
                    queries[destBlock].pushBack(table);
                }
            }
        }

        while (computeFinishedFlag() == 0) {
            for (int block = 0; block < nBlocks; block++) {
                newQueries[block] = new DataCollection(context);
            }

            for (int block = 0; block < nBlocks; block++) {
                DistributedStep8Local step8 = new DistributedStep8Local(context, Float.class, Method.defaultDense, block, nBlocks);
                step8.input.set(Step8LocalNumericTableInputId.step8InputClusterStructure, clusterStructure[block]);
                step8.input.set(Step8LocalNumericTableInputId.step8InputNClusters, nClusters[block]);
                step8.input.set(Step8LocalCollectionInputId.step8PartialQueries, queries[block]);
                DistributedPartialResultStep8 partialResult = step8.compute();

                clusterStructure[block] = partialResult.get(DistributedPartialResultStep8NumericTableId.step8ClusterStructure);
                finishedFlag[block] = partialResult.get(DistributedPartialResultStep8NumericTableId.step8FinishedFlag);
                nClusters[block] = partialResult.get(DistributedPartialResultStep8NumericTableId.step8NClusters);

                DataCollection curQueries = partialResult.get(DistributedPartialResultStep8CollectionId.step8Queries);

                for (int destBlock = 0; destBlock < nBlocks; destBlock++) {
                    NumericTable table = (NumericTable)curQueries.get(destBlock);
                    if (table.getNumberOfRows() > 0) {
                        newQueries[destBlock].pushBack(table);
                    }
                }
            }

            for (int block = 0; block < nBlocks; block++) {
                queries[block] = newQueries[block];
            }
        }

        {
            DataCollection partialNClusters = new DataCollection(context);
            for (int block = 0; block < nBlocks; block++) {
                partialNClusters.pushBack(nClusters[block]);
            }

            DistributedStep9Master step9 = new DistributedStep9Master(context, Float.class, Method.defaultDense);
            step9.input.set(Step9MasterCollectionInputId.partialNClusters, partialNClusters);
            DistributedPartialResultStep9 partialResult = step9.compute();
            DistributedResultStep9 result = step9.finalizeCompute();

            totalNClusters = result.get(DistributedResultStep9Id.step9NClusters);

            DataCollection curClusterOffsets = partialResult.get(DistributedPartialResultStep9Id.clusterOffsets);

            for (int block = 0; block < nBlocks; block++) {
                clusterOffset[block] = (NumericTable)curClusterOffsets.get(block);
            }
        }

        for (int block = 0; block < nBlocks; block++) {
            queries[block] = new DataCollection(context);
        }

        for (int block = 0; block < nBlocks; block++) {
            DistributedStep10Local step10 = new DistributedStep10Local(context, Float.class, Method.defaultDense, block, nBlocks);
            step10.input.set(Step10LocalNumericTableInputId.step10InputClusterStructure, clusterStructure[block]);
            step10.input.set(Step10LocalNumericTableInputId.step10ClusterOffset, clusterOffset[block]);
            DistributedPartialResultStep10 partialResult = step10.compute();

            clusterStructure[block] = partialResult.get(DistributedPartialResultStep10NumericTableId.step10ClusterStructure);
            finishedFlag[block] = partialResult.get(DistributedPartialResultStep10NumericTableId.step10FinishedFlag);

            DataCollection curQueries = partialResult.get(DistributedPartialResultStep10CollectionId.step10Queries);

            for (int destBlock = 0; destBlock < nBlocks; destBlock++) {
                NumericTable table = (NumericTable)curQueries.get(destBlock);
                if (table.getNumberOfRows() > 0) {
                    queries[destBlock].pushBack(table);
                }
            }
        }

        while (computeFinishedFlag() == 0) {
            for (int block = 0; block < nBlocks; block++) {
                newQueries[block] = new DataCollection(context);
            }

            for (int block = 0; block < nBlocks; block++) {
                DistributedStep11Local step11 = new DistributedStep11Local(context, Float.class, Method.defaultDense, block, nBlocks);
                step11.input.set(Step11LocalNumericTableInputId.step11InputClusterStructure, clusterStructure[block]);
                step11.input.set(Step11LocalCollectionInputId.step11PartialQueries, queries[block]);
                DistributedPartialResultStep11 partialResult = step11.compute();

                clusterStructure[block] = partialResult.get(DistributedPartialResultStep11NumericTableId.step11ClusterStructure);
                finishedFlag[block] = partialResult.get(DistributedPartialResultStep11NumericTableId.step11FinishedFlag);

                DataCollection curQueries = partialResult.get(DistributedPartialResultStep11CollectionId.step11Queries);

                for (int destBlock = 0; destBlock < nBlocks; destBlock++) {
                    NumericTable table = (NumericTable)curQueries.get(destBlock);
                    if (table.getNumberOfRows() > 0) {
                        newQueries[destBlock].pushBack(table);
                    }
                }
            }

            for (int block = 0; block < nBlocks; block++) {
                queries[block] = newQueries[block];
            }
        }

        for (int block = 0; block < nBlocks; block++) {
            assignmentQueries[block] = new DataCollection(context);
        }

        for (int block = 0; block < nBlocks; block++)
        {
            DistributedStep12Local step12 = new DistributedStep12Local(context, Float.class, Method.defaultDense, block, nBlocks);
            step12.input.set(Step12LocalNumericTableInputId.step12InputClusterStructure, clusterStructure[block]);
            step12.input.set(Step12LocalCollectionInputId.step12PartialOrders, partitionedPartialOrders[block]);
            DistributedPartialResultStep12 partialResult = step12.compute();

            DataCollection curAssignmentQueries = partialResult.get(DistributedPartialResultStep12Id.assignmentQueries);

            for (int destBlock = 0; destBlock < nBlocks; destBlock++) {
                NumericTable table = (NumericTable)curAssignmentQueries.get(destBlock);
                if (table.getNumberOfRows() > 0) {
                    assignmentQueries[destBlock].pushBack(table);
                }
            }
        }

        for (int block = 0; block < nBlocks; block++) {
            DistributedStep13Local step13 = new DistributedStep13Local(context, Float.class, Method.defaultDense);
            step13.input.set(Step13LocalCollectionInputId.partialAssignmentQueries, assignmentQueries[block]);
            step13.compute();
            DistributedResultStep13 result = step13.finalizeCompute();

            assignments[block] = result.get(DistributedResultStep13Id.step13Assignments);
        }
    }

    private static int computeFinishedFlag() {
        DataCollection partialFinishedFlags = new DataCollection(context);

        for (int block = 0; block < nBlocks; block++) {
            partialFinishedFlags.pushBack(finishedFlag[block]);
        }

        DistributedStep7Master step7 = new DistributedStep7Master(context, Float.class, Method.defaultDense);
        step7.input.set(Step7MasterCollectionInputId.partialFinishedFlags, partialFinishedFlags);
        DistributedPartialResultStep7 partialResult = step7.compute();
        NumericTable finishedFlag = partialResult.get(DistributedPartialResultStep7Id.finishedFlag);

        int finishedFlagValue = finishedFlag.getIntValue(0, 0);
        return finishedFlagValue;
    }

    private static void readData(int iNode) throws java.io.FileNotFoundException, java.io.IOException {

        /* Read dataset from a file and create a numeric table for storing the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset[iNode],
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();
        dataTable[iNode] = dataSource.getNumericTable();
    }


    private static void printResults() {
        Service.printNumericTable("Number of clusters:", totalNClusters);
        for (int block = 0; block < nBlocks; block++) {
            Service.printNumericTable("Assignments of first 20 observations from block:", assignments[block], 20);
        }
    }
}
