/* file: GbtRegDenseDistr.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 //     Java example of gradient boosted trees regression.
 //
 //     The program trains the gradient boosted trees regression model on a supplied
 //     training data set and then predicts previously unseen data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-GBTREGDENSEDISTR">
 * @example GbtRegDenseDistr.java
 */

package com.intel.daal.examples.gbt;

import com.intel.daal.algorithms.gbt.regression.*;
import com.intel.daal.algorithms.gbt.regression.prediction.*;
import com.intel.daal.algorithms.gbt.regression.training.*;
import com.intel.daal.algorithms.gbt.regression.init.*;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class GbtRegDenseDistr {
    /* Input data set parameters */
    private static final int nBlocks = 4;

    private static final String[] trainDataset = {
            "../data/distributed/df_regression_train_1.csv",
            "../data/distributed/df_regression_train_2.csv",
            "../data/distributed/df_regression_train_3.csv",
            "../data/distributed/df_regression_train_4.csv"
            };

    private static final String[] testDataset = {
            "../data/distributed/df_regression_test_1.csv",
            "../data/distributed/df_regression_test_2.csv",
            "../data/distributed/df_regression_test_3.csv",
            "../data/distributed/df_regression_test_4.csv"};

    private static final int nFeatures = 13;
    private static final int maxIterations = 40;
    private static final int maxBins = 66000;
    private static final int minBinSize = 5;

    private static NumericTable[] trainData = new NumericTable[nBlocks];
    private static NumericTable[] trainDependentVariable = new NumericTable[nBlocks];
    private static NumericTable[] testData = new NumericTable[nBlocks];
    private static NumericTable[] testGroundTruth = new NumericTable[nBlocks];

    private static NumericTable[]   binnedData = new NumericTable[nBlocks];
    private static NumericTable[]   transposedBinnedData = new NumericTable[nBlocks];
    private static NumericTable[]   dependentVariable = new NumericTable[nBlocks];
    private static NumericTable[]   initialResponse = new NumericTable[nBlocks];
    private static NumericTable[]   binSizes = new NumericTable[nBlocks];
    private static DataCollection[] binValues = new DataCollection[nBlocks];
    private static NumericTable[]   treeStructure = new NumericTable[nBlocks];
    private static NumericTable[]   treeOrder = new NumericTable[nBlocks];
    private static NumericTable[]   optCoeffs = new NumericTable[nBlocks];
    private static NumericTable[]   response = new NumericTable[nBlocks];

    private static DataCollection[] histograms = new DataCollection[nBlocks];
    private static DataCollection[] parentHistograms = new DataCollection[nBlocks];
    private static DataCollection[] totalHistograms = new DataCollection[nFeatures];
    private static DataCollection[] parentTotalHistograms = new DataCollection[nFeatures];

    private static DataCollection bestSplits;

    private static DataCollection[] finalizedTrees = new DataCollection[nBlocks];

    private static Model[] partialModel = new Model[nBlocks];
    private static NumericTable[] prediction = new NumericTable[nBlocks];

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        for (int iNode = 0; iNode < nBlocks; iNode++) {
            readTrainData(iNode);
            readTestData(iNode);
        }

        init();
        System.out.println("1");
        trainModel();

        System.out.println("2");
        testModel();

        System.out.println("3");
        printResults();

        System.out.println("4");
        context.dispose();
    }

    private static void init() {
        DataCollection meanDependentVariable = new DataCollection(context);
        DataCollection numberOfRows = new DataCollection(context);
        DataCollection binBorders = new DataCollection(context);
        DataCollection localBinSizes = new DataCollection(context);

        for (int i = 0; i < nBlocks; i++) {
            InitDistributedStep1Local step1 = new InitDistributedStep1Local(context, Float.class, InitMethod.defaultDense, maxBins);
            step1.input.set(InitStep1LocalNumericTableInputId.step1LocalData, trainData[i]);
            step1.input.set(InitStep1LocalNumericTableInputId.step1LocalDependentVariables, trainDependentVariable[i]);

            InitDistributedPartialResultStep1 partialResult = step1.compute();

            meanDependentVariable.pushBack(partialResult.get(InitDistributedPartialResultStep1Id.step1MeanDependentVariable));
            numberOfRows.pushBack(partialResult.get(InitDistributedPartialResultStep1Id.step1NumberOfRows));
            binBorders.pushBack(partialResult.get(InitDistributedPartialResultStep1Id.step1BinBorders));
            localBinSizes.pushBack(partialResult.get(InitDistributedPartialResultStep1Id.step1BinSizes));
        }

        NumericTable mergedBinBorders;

        {
            InitDistributedStep2Master step2 = new InitDistributedStep2Master(context, Float.class, InitMethod.defaultDense, maxBins, minBinSize);

            step2.input.set(InitStep2MasterCollectionInputId.step2MeanDependentVariable, meanDependentVariable);
            step2.input.set(InitStep2MasterCollectionInputId.step2NumberOfRows, numberOfRows);
            step2.input.set(InitStep2MasterCollectionInputId.step2BinBorders, binBorders);
            step2.input.set(InitStep2MasterCollectionInputId.step2BinSizes, localBinSizes);

            InitDistributedPartialResultStep2 partialResult = step2.compute();

            for (int i = 0; i < nBlocks; i++) {
                initialResponse[i] = partialResult.get(InitDistributedPartialResultStep2NumericTableId.step2InitialResponse);
                binValues[i] = partialResult.get(InitDistributedPartialResultStep2CollectionId.step2BinValues);
                binSizes[i] = partialResult.get(InitDistributedPartialResultStep2NumericTableId.step2BinQuantities);
            }

            mergedBinBorders = partialResult.get(InitDistributedPartialResultStep2NumericTableId.step2MergedBinBorders);
        }

        for (int i = 0; i < nBlocks; i++) {
            InitDistributedStep3Local step3 = new InitDistributedStep3Local(context, Float.class, InitMethod.defaultDense, maxBins);

            step3.input.set(InitStep3LocalNumericTableInputId.step3MergedBinBorders, mergedBinBorders);
            step3.input.set(InitStep3LocalNumericTableInputId.step3BinQuantities, binSizes[i]);
            step3.input.set(InitStep3LocalNumericTableInputId.step3LocalData, trainData[i]);
            step3.input.set(InitStep3LocalNumericTableInputId.step3InitialResponse, initialResponse[i]);

            InitDistributedPartialResultStep3 partialResult = step3.compute();

            binnedData[i] = partialResult.get(InitDistributedPartialResultStep3Id.step3BinnedData);
            transposedBinnedData[i] = partialResult.get(InitDistributedPartialResultStep3Id.step3TransposedBinnedData);
            response[i] = partialResult.get(InitDistributedPartialResultStep3Id.step3Response);
            treeOrder[i] = partialResult.get(InitDistributedPartialResultStep3Id.step3TreeOrder);
        }
    }

    private static void trainModel() {
        for (int i = 0; i < nBlocks; i++) {
            finalizedTrees[i] = new DataCollection(context);
        }

        for (int iter = 0; iter < maxIterations + 1; iter++) {
            // 1-st step "Update gradients local"
            for (int i = 0; i < nBlocks; i++) {
                DistributedStep1Local step1 = new DistributedStep1Local(context, Float.class, Method.defaultDense);
                step1.input.set(Step1LocalNumericTableInputId.step1BinnedData, binnedData[i]);
                step1.input.set(Step1LocalNumericTableInputId.step1DependentVariable, trainDependentVariable[i]);
                step1.input.set(Step1LocalNumericTableInputId.step1InputResponse, response[i]);
                if (iter > 0) {
                    step1.input.set(Step1LocalNumericTableInputId.step1InputTreeStructure, treeStructure[i]);
                }
                step1.input.set(Step1LocalNumericTableInputId.step1InputTreeOrder, treeOrder[i]);

                DistributedPartialResultStep1 partialResult = step1.compute();

                response[i] = partialResult.get(DistributedPartialResultStep1Id.response);
                optCoeffs[i] = partialResult.get(DistributedPartialResultStep1Id.optCoeffs);
                treeOrder[i] = partialResult.get(DistributedPartialResultStep1Id.treeOrder);
                treeStructure[i] = partialResult.get(DistributedPartialResultStep1Id.step1TreeStructure);

                if (iter > 0) {
                    finalizedTrees[i].pushBack(partialResult.get(DistributedPartialResultStep1Id.finalizedTree));
                }
            }

            if (iter == maxIterations) {
                break;
            }

            for (int j = 0; j < nBlocks; j++) {
                parentHistograms[j] = new DataCollection(context);
            }
            for (int j = 0; j < nFeatures; j++) {
                parentTotalHistograms[j] = new DataCollection(context);
            }

            while (computeFinishedFlag(treeStructure[0]) == 0) {
                // 3-rd step "Compute histograms local"
                for (int i = 0; i < nBlocks; i++) {
                    DistributedStep3Local step3 = new DistributedStep3Local(context, Float.class, Method.defaultDense);

                    step3.input.set(Step3LocalNumericTableInputId.step3BinnedData, binnedData[i]);
                    step3.input.set(Step3LocalNumericTableInputId.step3BinSizes, binSizes[i]);
                    step3.input.set(Step3LocalNumericTableInputId.step3InputTreeStructure, treeStructure[i]);
                    step3.input.set(Step3LocalNumericTableInputId.step3InputTreeOrder, treeOrder[i]);
                    step3.input.set(Step3LocalNumericTableInputId.step3OptCoeffs, optCoeffs[i]);
                    if (parentHistograms[i].size() > 0) {
                        step3.input.set(Step3LocalCollectionInputId.step3ParentHistograms, parentHistograms[i]);
                    }

                    DistributedPartialResultStep3 partialResult = step3.compute();

                    histograms[i] = partialResult.get(DistributedPartialResultStep3Id.histograms);
                }

                DataCollection[] histogramsForFeature = new DataCollection[nFeatures];

                for (int i = 0; i < nFeatures; i++) {
                    histogramsForFeature[i] = new DataCollection(context);
                    for (int j = 0; j < nBlocks; j++) {
                        DataCollection histogramsForBlocks = new DataCollection(context);
                        for (int k = 0; k < histograms[j].size(); k++) {
                            histogramsForBlocks.pushBack((NumericTable)(((DataCollection)(histograms[j].get(k))).get(i)));
                        }
                        histogramsForFeature[i].pushBack(histogramsForBlocks);
                    }
                }
                
                System.out.println("Before 4");
                DataCollection bestSplits = new DataCollection(context);

                int processedFeatures = 0;

                for (int i = 0; i < nBlocks; i++) {
                    // // System.out.print(i);
                    // // System.out.println(")");

                    // DistributedStep4Local step4 = new DistributedStep4Local(context, Float.class, Method.defaultDense);
                    // step4.input.set(Step4LocalNumericTableInputId.step4InputTreeStructure, treeStructure[0]);
                    // DataCollection featureIndices = new DataCollection(context);
                    // featureIndices.pushBack(new HomogenNumericTable(context, Integer.class, 1, 1, NumericTable.AllocationFlag.DoAllocate, (int)i));
                    // step4.input.set(Step4LocalCollectionInputId.step4FeatureIndices, featureIndices);
                    // // step4.input.set(Step4LocalNumericTableInputId.step4FeatureIndex, new HomogenNumericTable(context, Integer.class, 1, 1, NumericTable.AllocationFlag.DoAllocate, (int)i));
                    // // if (parentTotalHistograms[i].size() > 0)
                    // {
                    //     step4.input.set(Step4LocalCollectionInputId.step4ParentTotalHistograms, parentTotalHistograms[i]);
                    // }
                    // step4.input.set(Step4LocalCollectionInputId.step4PartialHistograms, histogramsForFeature[i]);

                    // System.out.println(step4.input.get(Step4LocalNumericTableInputId.step4InputTreeStructure));
                    // System.out.println(step4.input.get(Step4LocalCollectionInputId.step4FeatureIndices));
                    // System.out.println(step4.input.get(Step4LocalCollectionInputId.step4ParentTotalHistograms));
                    // System.out.println(step4.input.get(Step4LocalCollectionInputId.step4PartialHistograms));

                    // System.out.println("1");
                    // DistributedPartialResultStep4 partialResult = step4.compute();
                    // System.out.println("2");

                    // totalHistograms[i] = partialResult.get(DistributedPartialResultStep4Id.totalHistograms);
                    // bestSplits.pushBack(partialResult.get(DistributedPartialResultStep4Id.bestSplits));


                    int featuresForBlock = (nFeatures - processedFeatures) / (nBlocks - i);
                    System.out.print("featuresForBlock = ");
                    System.out.println(featuresForBlock);

                    DataCollection featureIndices = new DataCollection(context);
                    DataCollection parentTotalHistogramsForFeatures = new DataCollection(context);
                    DataCollection partialHistogramsForFeatures = new DataCollection(context);

                    for (int j = 0; j < featuresForBlock; j++) {
                        featureIndices.pushBack(new HomogenNumericTable(context, Integer.class, 1, 1, NumericTable.AllocationFlag.DoAllocate, (int)(processedFeatures+j)));
                        // if (parentTotalHistograms[i].size() > 0) {
                            parentTotalHistogramsForFeatures.pushBack(parentTotalHistograms[processedFeatures+j]);
                        // }
                        partialHistogramsForFeatures.pushBack(histogramsForFeature[processedFeatures+j]);
                    }
                    
                    DataCollection tmp = (DataCollection)(parentTotalHistogramsForFeatures.get(0));
                    System.out.print("tmp.size() = ");
                    System.out.println(tmp.size());
                    // Service.printNumericTable("parentTotalHistogramsForFeatures[0]", (NumericTable)(tmp.get(0)), 10);
                    // Service.printNumericTable("parentTotalHistogramsForFeatures[0]", (NumericTable)(tmp.get(1)), 10);
                    // Service.printNumericTable("parentTotalHistogramsForFeatures[0]", (NumericTable)(tmp.get(2)), 10);

                    DistributedStep4Local step4 = new DistributedStep4Local(context, Float.class, Method.defaultDense);

                    step4.input.set(Step4LocalNumericTableInputId.step4InputTreeStructure, treeStructure[0]);
                    step4.input.set(Step4LocalCollectionInputId.step4FeatureIndices, featureIndices);
                    // if (tmp.size() > 0) {
                        step4.input.set(Step4LocalCollectionInputId.step4ParentTotalHistograms, parentTotalHistogramsForFeatures);
                    // }
                    step4.input.set(Step4LocalCollectionInputId.step4PartialHistograms, partialHistogramsForFeatures);

                    DistributedPartialResultStep4 partialResult = step4.compute();

                    DataCollection totalHistogramsForFeatures = partialResult.get(DistributedPartialResultStep4Id.totalHistograms);
                    DataCollection bestSplitsForFeatures = partialResult.get(DistributedPartialResultStep4Id.bestSplits);

                    for (int j = 0; j < featuresForBlock; j++) {
                        totalHistograms[processedFeatures + j] = (DataCollection)(totalHistogramsForFeatures.get(j));
                        bestSplits.pushBack((DataCollection)(totalHistogramsForFeatures.get(j)));
                    }

                    processedFeatures += featuresForBlock;
                }
                System.out.println("After 4");

                for (int i = 0; i < nBlocks; i++) {
                    DistributedStep5Local step5 = new DistributedStep5Local(context, Float.class, Method.defaultDense);
                    step5.input.set(Step5LocalNumericTableInputId.step5BinnedData, binnedData[i]);
                    step5.input.set(Step5LocalNumericTableInputId.step5TransposedBinnedData, transposedBinnedData[i]);
                    step5.input.set(Step5LocalNumericTableInputId.step5BinSizes, binSizes[i]);
                    step5.input.set(Step5LocalNumericTableInputId.step5InputTreeStructure, treeStructure[i]);
                    step5.input.set(Step5LocalNumericTableInputId.step5InputTreeOrder, treeOrder[i]);
                    step5.input.set(Step5LocalCollectionInputId.step5PartialBestSplits, bestSplits);

                    DistributedPartialResultStep5 partialResult = step5.compute();

                    treeStructure[i] = partialResult.get(DistributedPartialResultStep5Id.step5TreeStructure);
                    treeOrder[i] = partialResult.get(DistributedPartialResultStep5Id.step5TreeOrder);
                }
                System.out.println("After 5");

                for (int j = 0; j < nBlocks; j++) {
                    parentHistograms[j] = histograms[j];
                }
                for (int j = 0; j < nFeatures; j++) {
                    parentTotalHistograms[j] = totalHistograms[j];
                }
            }
        }

        for (int i = 0; i < nBlocks; i++) {
            DistributedStep6Local step6 = new DistributedStep6Local(context, Float.class, Method.defaultDense);
            step6.input.set(Step6LocalNumericTableInputId.step6InitialResponse, initialResponse[i]);
            step6.input.set(Step6LocalCollectionInputId.step6BinValues, binValues[i]);
            step6.input.set(Step6LocalCollectionInputId.step6FinalizedTrees, finalizedTrees[i]);

            DistributedPartialResultStep6 partialResult = step6.compute();

            partialModel[i] = partialResult.get(DistributedPartialResultStep6Id.partialModel);
        }
    }

    private static int computeFinishedFlag(NumericTable treeStructure) {
        DistributedStep2Local step2 = new DistributedStep2Local(context, Float.class, Method.defaultDense);
        step2.input.set(Step2LocalNumericTableInputId.step2InputTreeStructure, treeStructure);
        DistributedPartialResultStep2 partialResult = step2.compute();
        NumericTable finishedFlag = partialResult.get(DistributedPartialResultStep2Id.finishedFlag);

        int finishedFlagValue = finishedFlag.getIntValue(0, 0);
        return finishedFlagValue;
    }


    private static void testModel() {
        for (int i = 0; i < nBlocks; i++) {
            /* Create algorithm objects for gradient boosted trees regression prediction with the fast method */
            PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

            /* Pass a testing data set and the trained model to the algorithm */
            algorithm.input.set(NumericTableInputId.data, testData[i]);
            algorithm.input.set(ModelInputId.model, partialModel[i]);

            /* Compute prediction results */
            PredictionResult result = algorithm.compute();
            prediction[i] = result.get(PredictionResultId.prediction);
        }
    }

    private static void readTrainData(int iNode) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the data from the input data sets */
        FileDataSource trainDataSource = new FileDataSource(context, trainDataset[iNode],
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        trainData[iNode] = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        trainDependentVariable[iNode] = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);

        /* Create Numeric Tables for training data and labels */
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData[iNode]);
        mergedData.addNumericTable(trainDependentVariable[iNode]);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);
    }

    private static void readTestData(int iNode) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the data from the input data sets */
        FileDataSource testDataSource = new FileDataSource(context, testDataset[iNode],
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        testData[iNode] = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testGroundTruth[iNode] = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);

        /* Create Numeric Tables for training data and labels */
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData[iNode]);
        mergedData.addNumericTable(testGroundTruth[iNode]);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);
    }

    private static void printResults() {
        for (int i = 0; i < nBlocks; i++) {
            Service.printNumericTable("Gragient boosted trees prediction results (first 10 rows):", prediction[i], 10);
            Service.printNumericTable("Ground truth (first 10 rows):", testGroundTruth[i], 10);
        }
    }
}
