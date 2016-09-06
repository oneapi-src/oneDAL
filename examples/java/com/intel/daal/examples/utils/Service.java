/* file: Service.java */
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
 //     Auxiliary functions used in Java examples
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;

import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.DaalContext;

public class Service {
    public static void readRow(String line, int offset, int nCols, double[] data) throws IOException {
        if (line == null) {
            throw new IOException("Unable to read input dataset");
        }

        String[] elements = line.split(",");
        for (int j = 0; j < nCols; j++) {
            data[offset + j] = Double.parseDouble(elements[j]);
        }
    }

    public static void readRow(String line, int offset, int nCols, long[] data) throws IOException {
        if (line == null) {
            throw new IOException("Unable to read input dataset");
        }

        String[] elements = line.split(",");
        for (int j = 0; j < nCols; j++) {
            data[offset + j] = Long.parseLong(elements[j]);
        }
    }

    public static void readSparseData(String dataset, int nVectors, int nNonZeroValues, long[] rowOffsets,
            long[] colIndices, double[] data) {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(dataset));
            readRow(bufferedReader.readLine(), 0, nVectors + 1, rowOffsets);
            readRow(bufferedReader.readLine(), 0, nNonZeroValues, colIndices);
            readRow(bufferedReader.readLine(), 0, nNonZeroValues, data);
            bufferedReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (NumberFormatException e) {
            e.printStackTrace();
        }
    }

    private static int getRowLength(String line) {
        String[] elements = line.split(",");
        return elements.length;
    }

    public static CSRNumericTable createSparseTable(DaalContext context, String dataset) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(dataset));

        String rowIndexLine = bufferedReader.readLine();
        int nVectors = getRowLength(rowIndexLine);
        long[] rowOffsets = new long[nVectors];

        readRow(rowIndexLine, 0, nVectors, rowOffsets);
        nVectors = nVectors - 1;

        String columnsLine = bufferedReader.readLine();
        int nCols = getRowLength(columnsLine);

        long[] colIndices = new long[nCols];
        readRow(columnsLine, 0, nCols, colIndices);

        String valuesLine = bufferedReader.readLine();
        int nNonZeros = getRowLength(valuesLine);

        double[] data = new double[nNonZeros];
        readRow(valuesLine, 0, nNonZeros, data);

        bufferedReader.close();

        long maxCol = 0;
        for (int i = 0; i < nCols; i++) {
            if (colIndices[i] > maxCol) {
                maxCol = colIndices[i];
            }
        }
        int nFeatures = (int) maxCol;

        if (nCols != nNonZeros || nNonZeros != (rowOffsets[nVectors] - 1) || nFeatures == 0 || nVectors == 0) {
            throw new IOException("Unable to read input dataset");
        }

        return new CSRNumericTable(context, data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    public static void printClassificationResult(double[] groundTruth, double[] classificationResults,
            String classificatorName) {
        System.out.println(classificatorName + " classification:");
        System.out.println("Ground truth | Classification results");

        for (int i = 0; i < Math.min(groundTruth.length, 20); i++) {
            System.out.format("%+f\t\t%+f\n", groundTruth[i], classificationResults[i]);
        }
    }

    public static void printClassificationResult(NumericTable groundTruth, NumericTable classificationResults,
            String header1, String header2, String message, int nMaxRows) {
        int nCols = (int) groundTruth.getNumberOfColumns();
        int nRows = Math.min((int) groundTruth.getNumberOfRows(), nMaxRows);

        DoubleBuffer dataGroundTruth = DoubleBuffer.allocate(nCols * nRows);
        dataGroundTruth = groundTruth.getBlockOfRows(0, nRows, dataGroundTruth);

        DoubleBuffer dataClassificationResults = DoubleBuffer.allocate(nCols * nRows);
        dataClassificationResults = classificationResults.getBlockOfRows(0, nRows, dataClassificationResults);

        System.out.println(message);
        System.out.println(header1 + "\t" + header2);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < 1; j++) {
                System.out.format("%+.0f\t\t%+.0f\n", dataGroundTruth.get(i * nCols + j),
                        dataClassificationResults.get(i * nCols + j));
            }
        }
    }

    public static void printClassificationResult(long[] groundTruth, long[] classificationResults,
            String classificatorName) {
        System.out.println(classificatorName + " classification:");
        System.out.println("Ground truth | Classification results");

        for (int i = 0; i < Math.min(groundTruth.length, 20); i++) {
            System.out.format("%+d\t\t%+d\n", groundTruth[i], classificationResults[i]);
        }
    }

    public static void printClassificationResult(long[] groundTruth, int[] classificationResults,
            String classificatorName) {
        System.out.println(classificatorName + " classification:");
        System.out.println("Ground truth | Classification results");

        for (int i = 0; i < Math.min(groundTruth.length, 20); i++) {
            System.out.format("%+d\t\t%+d\n", groundTruth[i], classificationResults[i]);
        }
    }

    public static void printMatrix(double[] matrix, int nCols, int nRows, String header) {
        System.out.println(header);
        DecimalFormat numberFormat = new DecimalFormat("##0.00");
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                System.out.print(numberFormat.format(matrix[i * nCols + j]) + "\t\t");
            }
            System.out.println();
        }
    }

    public static void printTriangularMatrix(double[] triangularMatrix, int nDimensions, String header) {
        int index = 0;
        for (int i = 0; i < nDimensions; i++) {
            for (int j = 0; j <= i; j++) {
                System.out.print(triangularMatrix[index++] + "   ");
            }
            System.out.println();
        }
    }

    public static void printPackedNumericTable(HomogenNumericTable nt, long nDimensions, String header) {
        double[] results = nt.getDoubleArray();
        printTriangularMatrix(results, (int) nDimensions, header);
    }

    public static void printNumericTable(String header, NumericTable nt, long nPrintedRows, long nPrintedCols) {
        long nNtCols = nt.getNumberOfColumns();
        long nNtRows = nt.getNumberOfRows();
        long nRows = nNtRows;
        long nCols = nNtCols;

        if (nPrintedRows > 0) {
            nRows = Math.min(nNtRows, nPrintedRows);
        }

        DoubleBuffer result = DoubleBuffer.allocate((int) (nNtCols * nRows));
        result = nt.getBlockOfRows(0, nRows, result);

        if (nPrintedCols > 0) {
            nCols = Math.min(nNtCols, nPrintedCols);
        }

        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n");
        for (long i = 0; i < nRows; i++) {
            for (long j = 0; j < nCols; j++) {
                String tmp = String.format("%-6.3f   ", result.get((int) (i * nNtCols + j)));
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }

    public static void printNumericTable(String header, CSRNumericTable nt, long nPrintedRows, long nPrintedCols) {
        long[] rowOffsets = nt.getRowOffsetsArray();
        long[] colIndices = nt.getColIndicesArray();
        double[] values = nt.getDoubleArray();

        long nNtCols = nt.getNumberOfColumns();
        long nNtRows = nt.getNumberOfRows();
        long nRows = nNtRows;
        long nCols = nNtCols;

        if (nPrintedRows > 0) {
            nRows = Math.min(nNtRows, nPrintedRows);
        }

        if (nPrintedCols > 0) {
            nCols = Math.min(nNtCols, nPrintedCols);
        }

        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n");

        double[] oneDenseRow = new double[(int) nCols];
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                oneDenseRow[j] = 0;
            }
            int nElementsInRow = (int) (rowOffsets[i + 1] - rowOffsets[i]);
            for (int k = 0; k < nElementsInRow; k++) {
                oneDenseRow[(int) (colIndices[(int) (rowOffsets[i] - 1 + k)] - 1)] = values[(int) (rowOffsets[i] - 1
                        + k)];
            }
            for (int j = 0; j < nCols; j++) {
                String tmp = String.format("%-6.3f   ", oneDenseRow[j]);
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }

    public static void printNumericTable(String header, NumericTable nt, long nRows) {
        printNumericTable(header, nt, nRows, nt.getNumberOfColumns());
    }

    public static void printNumericTable(String header, NumericTable nt) {
        printNumericTable(header, nt, nt.getNumberOfRows());
    }

    public static void printNumericTable(String header, CSRNumericTable nt, long nRows) {
        printNumericTable(header, nt, nRows, nt.getNumberOfColumns());
    }

    public static void printNumericTable(String header, CSRNumericTable nt) {
        printNumericTable(header, nt, nt.getNumberOfRows());
    }

    public static void printAprioriItemsets(HomogenNumericTable largeItemsetsTable,
            HomogenNumericTable largeItemsetsSupportTable) {
        /* Get sizes of tables to store large item sets */
        int nItemsInLargeItemsets = (int) largeItemsetsTable.getNumberOfRows();
        int largeItemsetCount = (int) largeItemsetsSupportTable.getNumberOfRows();
        int nItemsetToPrint = 20;

        /* Get item sets and their support values */
        IntBuffer bufLargeItemsets = IntBuffer
                .allocate(nItemsInLargeItemsets * (int) largeItemsetsTable.getNumberOfColumns());
        bufLargeItemsets = largeItemsetsTable.getBlockOfRows(0, nItemsInLargeItemsets, bufLargeItemsets);
        int[] largeItemsets = new int[bufLargeItemsets.capacity()];
        bufLargeItemsets.get(largeItemsets);

        IntBuffer bufLargeItemsetsSupportData = IntBuffer
                .allocate(largeItemsetCount * (int) largeItemsetsSupportTable.getNumberOfColumns());
        bufLargeItemsetsSupportData = largeItemsetsSupportTable.getBlockOfRows(0, largeItemsetCount,
                bufLargeItemsetsSupportData);
        int[] largeItemsetsSupportData = new int[bufLargeItemsetsSupportData.capacity()];
        bufLargeItemsetsSupportData.get(largeItemsetsSupportData);

        ArrayList<ArrayList<Integer>> largeItemsetsVector = new ArrayList<ArrayList<Integer>>(largeItemsetCount);

        for (int i = 0; i < largeItemsetCount; i++) {
            largeItemsetsVector.add(new ArrayList<Integer>());
        }

        for (int i = 0; i < nItemsInLargeItemsets; i++) {
            largeItemsetsVector.get(largeItemsets[2 * i]).add(largeItemsets[2 * i + 1]);
        }

        ArrayList<Integer> supportVector = new ArrayList<Integer>(largeItemsetCount);
        for (int i = 0; i < largeItemsetCount; i++) {
            supportVector.add(0);
        }

        for (int i = 0; i < largeItemsetCount; i++) {
            int index = largeItemsetsSupportData[2 * i];
            supportVector.set(index, largeItemsetsSupportData[2 * i + 1]);
        }

        System.out.println("Apriori example program results");

        System.out.println("Last " + nItemsetToPrint + " large itemsets: ");
        System.out.println("Itemset\t\t\tSupport");

        int iMin = ((largeItemsetCount > nItemsetToPrint) ? largeItemsetCount - nItemsetToPrint : 0);
        for (int i = iMin; i < largeItemsetCount; i++) {
            System.out.print("{");
            for (int l = 0; l < largeItemsetsVector.get(i).size() - 1; l++) {
                System.out.print(largeItemsetsVector.get(i).get(l) + ", ");
            }
            System.out.print(largeItemsetsVector.get(i).get(largeItemsetsVector.get(i).size() - 1) + "}\t\t");

            System.out.println(supportVector.get(i));
        }
    }

    public static void printAprioriRules(HomogenNumericTable leftItemsTable, HomogenNumericTable rightItemsTable,
            HomogenNumericTable confidenceTable) {
        int nRulesToPrint = 20;
        /* Get sizes of tables to store association rules */
        int nLeftItems = (int) leftItemsTable.getNumberOfRows();
        int nRightItems = (int) rightItemsTable.getNumberOfRows();
        int nRules = (int) confidenceTable.getNumberOfRows();

        /* Get association rules data */

        IntBuffer bufLeftItems = IntBuffer.allocate(nLeftItems * (int) leftItemsTable.getNumberOfColumns());
        bufLeftItems = leftItemsTable.getBlockOfRows(0, nLeftItems, bufLeftItems);
        int[] leftItems = new int[bufLeftItems.capacity()];
        bufLeftItems.get(leftItems);

        IntBuffer bufRightItems = IntBuffer.allocate(nRightItems * (int) rightItemsTable.getNumberOfColumns());
        bufRightItems = rightItemsTable.getBlockOfRows(0, nRightItems, bufRightItems);
        int[] rightItems = new int[bufRightItems.capacity()];
        bufRightItems.get(rightItems);

        double[] confidence = confidenceTable.getDoubleArray();

        ArrayList<ArrayList<Integer>> leftItemsVector = new ArrayList<ArrayList<Integer>>(nRules);
        for (int i = 0; i < nRules; i++) {
            leftItemsVector.add(new ArrayList<Integer>());
        }

        if (nRules == 0) {
            System.out.println("No association rules were found ");
            return;
        }

        for (int i = 0; i < nLeftItems; i++) {
            leftItemsVector.get((leftItems[2 * i])).add(leftItems[2 * i + 1]);
        }

        ArrayList<ArrayList<Integer>> rightItemsVector = new ArrayList<ArrayList<Integer>>(nRules);
        for (int i = 0; i < nRules; i++) {
            rightItemsVector.add(new ArrayList<Integer>());
        }

        for (int i = 0; i < nRightItems; i++) {
            rightItemsVector.get((rightItems[2 * i])).add(rightItems[2 * i + 1]);
        }

        ArrayList<Double> confidenceVector = new ArrayList<Double>(nRules);
        for (int i = 0; i < nRules; i++) {
            confidenceVector.add(confidence[i]);
        }

        System.out.println("Last " + nRulesToPrint + " association rules: ");
        System.out.println("Rule" + "\t\t\t\tConfidence");

        int iMin = ((nRules > nRulesToPrint) ? (nRules - nRulesToPrint) : 0);
        for (int i = iMin; i < nRules; i++) {
            System.out.print("{");
            for (int l = 0; l < leftItemsVector.get(i).size() - 1; l++) {
                System.out.print(leftItemsVector.get(i).get(l) + ", ");
            }
            System.out.print(leftItemsVector.get(i).get(leftItemsVector.get(i).size() - 1) + "} => {");

            for (int l = 0; l < rightItemsVector.get(i).size() - 1; l++) {
                System.out.print(rightItemsVector.get(i).get(l) + ", ");
            }
            System.out.print(rightItemsVector.get(i).get(rightItemsVector.get(i).size() - 1) + "}\t\t");

            System.out.println(confidenceVector.get(i));
        }
    }

    public static void computePartialModelBlocksToNode(DaalContext context, int nNodes,
                                                CSRNumericTable[] dataTable, CSRNumericTable[] dataTableTransposed,
                                                long[] usersPartition, long[] itemsPartition,
                                                KeyValueDataCollection[] usersOutBlocks,
                                                KeyValueDataCollection[] itemsOutBlocks)
    {
        usersPartition[0] = 0;
        itemsPartition[0] = 0;
        for (int i = 0; i < nNodes; i++)
        {
            usersPartition[i + 1] = usersPartition[i] + dataTable[i].getNumberOfRows();
            itemsPartition[i + 1] = itemsPartition[i] + dataTableTransposed[i].getNumberOfRows();
        }

        for (int i = 0; i < nNodes; i++)
        {
            usersOutBlocks[i] = computeOutBlocks(context, nNodes, dataTable[i],           itemsPartition);
            itemsOutBlocks[i] = computeOutBlocks(context, nNodes, dataTableTransposed[i], usersPartition);
        }
    }

    private static KeyValueDataCollection computeOutBlocks(DaalContext context,
                                                           int nNodes, CSRNumericTable dataBlock,
                                                           long[] dataBlockPartition) {
        long nRows = dataBlock.getNumberOfRows();
        int iNRows = (int)nRows;
        boolean[] blockIdFlags = new boolean[iNRows * nNodes];
        for (int i = 0; i < iNRows * nNodes; i++) {
            blockIdFlags[i] = false;
        }

        long[] rowOffsets = dataBlock.getRowOffsetsArray();
        long[] colIndices = dataBlock.getColIndicesArray();

        for (long i = 0; i < nRows; i++) {
            for (long j = rowOffsets[(int)i] - 1; j < rowOffsets[(int)i+1] - 1; j++) {
                for (int k = 1; k < nNodes + 1; k++) {
                    if (dataBlockPartition[k-1] <= colIndices[(int)j] - 1 && colIndices[(int)j] - 1 < dataBlockPartition[k]) {
                        blockIdFlags[(k-1) * iNRows + (int)i] = true;
                    }
                }
            }
        }

        long[] nNotNull = new long[nNodes];
        for (int i = 0; i < nNodes; i++) {
            nNotNull[i] = 0;
            for (int j = 0; j < iNRows; j++) {
                if (blockIdFlags[i * iNRows + j]) {
                    nNotNull[i] += 1;
                }
            }
        }
        KeyValueDataCollection result = new KeyValueDataCollection(context);

        for (int i = 0; i < nNodes; i++) {
            HomogenNumericTable indicesTable = new HomogenNumericTable(context, Integer.class, 1, nNotNull[i],
                                                                       NumericTable.AllocationFlag.DoAllocate);
            IntBuffer indicesBuffer = IntBuffer.allocate((int)nNotNull[i]);
            indicesBuffer = indicesTable.getBlockOfRows(0, nNotNull[i], indicesBuffer);
            int indexId = 0;
            for (int j = 0; j < iNRows; j++) {
                if (blockIdFlags[i * iNRows + j]) {
                    indicesBuffer.put(indexId, j);
                    indexId++;
                }
            }
            indicesTable.releaseBlockOfRows(0, nNotNull[i], indicesBuffer);
            result.set(i, indicesTable);
        }
        return result;
    }

    public static void printALSRatings(long usersOffset, long itemsOffset,
                                       NumericTable ratings) {
        long nUsers = ratings.getNumberOfRows();
        long nItems = ratings.getNumberOfColumns();

        double[] ratingsData = ((HomogenNumericTable)ratings).getDoubleArray();

        System.out.println(" User ID, Item ID, rating");
        for (long i = 0; i < nUsers; i++) {
            for (long j = 0; j < nItems; j++) {
                long userId = i + usersOffset;
                long itemId = j + itemsOffset;
                System.out.println(userId + ", " + itemId + ", " + ratingsData[(int)(i * nItems + j)]);
            }
        }
    }

    public static void printTensor(String header, Tensor dataTensor, int nPrintedRows, int nPrintedCols) {
        long[] dims = dataTensor.getDimensions();
        int nRows = (int)dims[0];
        if (nPrintedRows == 0 || nRows < nPrintedRows) nPrintedRows = nRows;

        int nCols = 1;
        for (int i = 1; i < dims.length; i++) {
            nCols *= dims[i];
        }
        DoubleBuffer result = DoubleBuffer.allocate(nRows * nCols);
        long[] fixed = {};
        result = dataTensor.getSubtensor(fixed, 0, nPrintedRows, result);

        if (nPrintedCols == 0 || nCols < nPrintedCols)  {
            nPrintedCols = nCols;
        }

        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n");
        for (long i = 0; i < nPrintedRows; i++) {
            for (long j = 0; j < nPrintedCols; j++) {
                String tmp = String.format("%-6.3f   ", result.get((int) (i * nCols + j)));
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }

    public static void printTensors(String header1, String header2, String message,
                                    Tensor dataTensor1, Tensor dataTensor2, int nPrintedRows) {
        long[] dims1 = dataTensor1.getDimensions();
        int nRows1 = (int)dims1[0];
        if (nPrintedRows == 0 || nRows1 < nPrintedRows) nPrintedRows = nRows1;

        int nCols1 = 1;
        for (int i = 1; i < dims1.length; i++) {
            nCols1 *= dims1[i];
        }

        long[] dims2 = dataTensor2.getDimensions();
        int nRows2 = (int)dims2[0];
        if (nPrintedRows == 0 || nRows2 < nPrintedRows) nPrintedRows = nRows2;

        int nCols2 = 1;
        for (int i = 1; i < dims2.length; i++) {
            nCols2 *= dims2[i];
        }

        long[] fixed = {};

        DoubleBuffer result1 = DoubleBuffer.allocate(nRows1 * nCols1);
        result1 = dataTensor1.getSubtensor(fixed, 0, nPrintedRows, result1);

        DoubleBuffer result2 = DoubleBuffer.allocate(nRows2 * nCols2);
        result2 = dataTensor2.getSubtensor(fixed, 0, nPrintedRows, result2);

        StringBuilder builder = new StringBuilder();
        builder.append(message);
        builder.append("\n");
        builder.append(header1 + "\t" + header2 + "\n");
        for (long i = 0; i < nPrintedRows; i++) {
            for (long j = 0; j < nCols1; j++) {
                String tmp = String.format("%-6.3f   ", result1.get((int) (i * nCols1 + j)));
                builder.append(tmp);
            }
            builder.append("\t");
            for (long j = 0; j < nCols2; j++) {
                String tmp = String.format("%-6.3f   ", result2.get((int) (i * nCols2 + j)));
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }

    public static Tensor readTensorFromCSV(DaalContext context, String datasetFileName) {
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();
        NumericTable nt = dataSource.getNumericTable();
        int nRows = (int)nt.getNumberOfRows();
        int nCols = (int)nt.getNumberOfColumns();
        if (nCols > 1) {
            long[] dims = {nRows, nCols};
            float[] data = new float[nRows * nCols];
            DoubleBuffer buffer = DoubleBuffer.allocate(nRows * nCols);
            buffer = nt.getBlockOfRows(0, nRows, buffer);
            for (int i = 0; i < nRows * nCols; i++) {
                data[i] = (float)buffer.get(i);
            }

            return new HomogenTensor(context, dims, data);
        } else {
            long[] dims = {nRows};
            float[] data = new float[nRows];
            DoubleBuffer buffer = DoubleBuffer.allocate(nRows);
            buffer = nt.getBlockOfRows(0, nRows, buffer);
            for (int i = 0; i < nRows; i++) {
                data[i] = (float)buffer.get(i);
            }

            return new HomogenTensor(context, dims, data);
        }
    }
}
