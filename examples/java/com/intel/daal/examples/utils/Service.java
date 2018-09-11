/* file: Service.java */
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
 //     Auxiliary functions used in Java examples
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
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

    public static void readRow(String line, int offset, int nCols, float[] data) throws IOException {
        if (line == null) {
            throw new IOException("Unable to read input dataset");
        }

        String[] elements = line.split(",");
        for (int j = 0; j < nCols; j++) {
            data[offset + j] = Float.parseFloat(elements[j]);
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

        float[] data = new float[nNonZeros];
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

    public static void printClassificationResult(float[] groundTruth, float[] classificationResults,
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

        FloatBuffer dataGroundTruth = FloatBuffer.allocate(nCols * nRows);
        dataGroundTruth = groundTruth.getBlockOfRows(0, nRows, dataGroundTruth);

        FloatBuffer dataClassificationResults = FloatBuffer.allocate(nCols * nRows);
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

    public static boolean isUpper(NumericTable.StorageLayout layout)
    {
        if (layout.ordinal() == NumericTable.StorageLayout.upperPackedSymmetricMatrix.ordinal()  ||
            layout.ordinal() == NumericTable.StorageLayout.upperPackedTriangularMatrix.ordinal())
        {
            return true;
        }
        return false;
    }

    public static boolean isLower(NumericTable.StorageLayout layout)
    {
        if (layout.ordinal() == NumericTable.StorageLayout.lowerPackedSymmetricMatrix.ordinal()  ||
            layout.ordinal() == NumericTable.StorageLayout.lowerPackedTriangularMatrix.ordinal())
        {
            return true;
        }
        return false;
    }

    public static void printNumericTable(String header, NumericTable nt, long nPrintedRows, long nPrintedCols) {
        long nNtCols = nt.getNumberOfColumns();
        long nNtRows = nt.getNumberOfRows();
        long nRows = nNtRows;
        long nCols = nNtCols;

        NumericTable.StorageLayout layout = nt.getDataLayout();

        if (nPrintedRows > 0) {
            nRows = Math.min(nNtRows, nPrintedRows);
        }

        FloatBuffer result = FloatBuffer.allocate((int) (nNtCols * nRows));
        result = nt.getBlockOfRows(0, nRows, result);

        if (nPrintedCols > 0) {
            nCols = Math.min(nNtCols, nPrintedCols);
        }

        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n");

        if( isLower(layout) )
        {
            for (long i = 0; i < nRows; i++) {
                for (long j = 0; j <= i; j++) {
                    String tmp = String.format("%-6.3f   ", result.get((int) (i * nNtCols + j)));
                    builder.append(tmp);
                }
                builder.append("\n");
            }
        }
        else if( isUpper(layout) )
        {

            for (long i = 0; i < nRows; i++) {
                for(int k=0; k < i; k++)
                        builder.append("         ");
                for (long j = i; j < nCols; j++) {
                    String tmp = String.format("%-6.3f   ", result.get((int) (i * nNtCols + j)));
                    builder.append(tmp);
                }
                builder.append("\n");
            }

        }
        else if( isLower(layout) != true && isUpper(layout) != true)
        {
            for (long i = 0; i < nRows; i++) {
                for (long j = 0; j < nCols; j++) {
                    String tmp = String.format("%-6.3f   ", result.get((int) (i * nNtCols + j)));
                    builder.append(tmp);
                }
                builder.append("\n");
            }
        }
        System.out.println(builder.toString());
    }

    public static void printNumericTable(String header, CSRNumericTable nt, long nPrintedRows, long nPrintedCols) {
        long[] rowOffsets = nt.getRowOffsetsArray();
        long[] colIndices = nt.getColIndicesArray();
        float[] values = nt.getFloatArray();

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

        float[] oneDenseRow = new float[(int) nCols];
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

    public static void printNumericTables(NumericTable dataTable1, NumericTable dataTable2,String title1, String title2 ,
                                            String message, long nPrintedRows)
    {
        long nRows1 = dataTable1.getNumberOfRows();
        long nRows2 = dataTable2.getNumberOfRows();
        long nCols1 = dataTable1.getNumberOfColumns();
        long nCols2 = dataTable2.getNumberOfColumns();

        long nRows = Math.min(nRows1, nRows2);
        if (nPrintedRows > 0)
        {
            nRows = Math.min(Math.min(nRows1, nRows2), nPrintedRows);
        }

        FloatBuffer result1 = FloatBuffer.allocate((int) (nCols1 * nRows));
        result1 = dataTable1.getBlockOfRows(0, nRows, result1);

        FloatBuffer result2 = FloatBuffer.allocate((int) (nCols2 * nRows));
        result2 = dataTable2.getBlockOfRows(0, nRows, result2);

        StringBuilder builder = new StringBuilder();
        builder.append(message);
        builder.append("\n");
        builder.append(title1);

        StringBuilder builderHelp = new StringBuilder();
        for (long j = 0; j < nCols1; j++) {
                String tmp = String.format("%-6.3f   ", result1.get((int) (0 * nCols1 + j)));
                builderHelp.append(tmp);
            }
        int interval = builderHelp.length() - title1.length();

        for(int i=0; i < interval; i++)
        {
            builder.append(" ");
        }
        builder.append("     ");
        builder.append(title2);
        builder.append("\n");

        for (long i = 0; i < nRows; i++) {
            for (long j = 0; j < nCols1; j++) {
                String tmp = String.format("%-6.3f   ", result1.get((int) (i * nCols1 + j)));
                builder.append(tmp);
            }
            builder.append("     ");
            for (long j = 0; j < nCols2; j++) {
                String tmp = String.format("%-6.3f   ", result2.get((int) (i * nCols2 + j)));
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
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

        System.out.println("\nApriori example program results");

        System.out.println("\nLast " + nItemsetToPrint + " large itemsets: ");
        System.out.println("\nItemset\t\t\tSupport");

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

        FloatBuffer bufConfidence = FloatBuffer.allocate(nRules * (int) confidenceTable.getNumberOfColumns());
        bufConfidence = confidenceTable.getBlockOfRows(0, nRules, bufConfidence);
        float[] confidence = new float[bufConfidence.capacity()];
        bufConfidence.get(confidence);

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

        ArrayList<Float> confidenceVector = new ArrayList<Float>(nRules);
        for (int i = 0; i < nRules; i++) {
            confidenceVector.add(confidence[i]);
        }

        System.out.println("\nLast " + nRulesToPrint + " association rules: ");
        System.out.println("\nRule" + "\t\t\t\tConfidence");

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

    public static void printALSRatings(NumericTable usersOffsetTable, NumericTable itemsOffsetTable,
                                       NumericTable ratings) {
        long nUsers = ratings.getNumberOfRows();
        long nItems = ratings.getNumberOfColumns();

        float[] ratingsData = ((HomogenNumericTable)ratings).getFloatArray();
        IntBuffer usersOffsetBuf = IntBuffer.allocate(1);
        IntBuffer itemsOffsetBuf = IntBuffer.allocate(1);
        usersOffsetBuf = usersOffsetTable.getBlockOfRows(0, 1, usersOffsetBuf);
        itemsOffsetBuf = itemsOffsetTable.getBlockOfRows(0, 1, itemsOffsetBuf);
        int[] usersOffsetData = new int[1];
        int[] itemsOffsetData = new int[1];
        usersOffsetBuf.get(usersOffsetData);
        itemsOffsetBuf.get(itemsOffsetData);
        long usersOffset = (long)usersOffsetData[0];
        long itemsOffset = (long)itemsOffsetData[0];

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
        FloatBuffer result = FloatBuffer.allocate(nRows * nCols);
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

    public static void printTensor3d(String header, Tensor dataTensor, int nFirstDim, int nSecondDim) {
        long[] dims = dataTensor.getDimensions();
        int nRows = (int)dims[0];
        int nCols = (int)dims[1];
        int nThirdDim = (int)dims[2];
        if(nFirstDim != 0)
        {
            nFirstDim = Math.min(nRows, nFirstDim);
        }
        else
        {
            nFirstDim = nRows;
        }

        if(nSecondDim != 0)
        {
            nSecondDim = Math.min(nCols, nSecondDim);
        }
        else
        {
            nSecondDim = nCols;
        }

        FloatBuffer result = FloatBuffer.allocate(nRows * nCols * nThirdDim);
        long[] fixed = {};
        result = dataTensor.getSubtensor(fixed, 0, nFirstDim, result);

        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n\n");

        for(int i = 0; i < nFirstDim; i++)
        {
            for(int j=0; j < nSecondDim; j++)
            {
                for(int k=0; k < nThirdDim; k++)
                {
                    String tmp = String.format("%-6.3f   ", result.get((int) (i * nThirdDim * nSecondDim + j*nThirdDim + k)));
                    builder.append(tmp);
                }
                builder.append("\n");

            }
            if(i != (nFirstDim - 1))
                builder.append("\n\n");

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

        FloatBuffer result1 = FloatBuffer.allocate(nRows1 * nCols1);
        result1 = dataTensor1.getSubtensor(fixed, 0, nPrintedRows, result1);

        FloatBuffer result2 = FloatBuffer.allocate(nRows2 * nCols2);
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

    public static Tensor readTensorFromCSV(DaalContext context, String datasetFileName, boolean allowOneColumn) {
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();
        NumericTable nt = dataSource.getNumericTable();
        int nRows = (int)nt.getNumberOfRows();
        int nCols = (int)nt.getNumberOfColumns();
        if (nCols > 1 || allowOneColumn) {
            long[] dims = {nRows, nCols};
            float[] data = new float[nRows * nCols];
            FloatBuffer buffer = FloatBuffer.allocate(nRows * nCols);
            buffer = nt.getBlockOfRows(0, nRows, buffer);
            for (int i = 0; i < nRows * nCols; i++) {
                data[i] = (float)buffer.get(i);
            }

            return new HomogenTensor(context, dims, data);
        } else {
            long[] dims = {nRows};
            float[] data = new float[nRows];
            FloatBuffer buffer = FloatBuffer.allocate(nRows);
            buffer = nt.getBlockOfRows(0, nRows, buffer);
            for (int i = 0; i < nRows; i++) {
                data[i] = (float)buffer.get(i);
            }

            return new HomogenTensor(context, dims, data);
        }
    }

    public static Tensor readTensorFromCSV(DaalContext context, String datasetFileName) {
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        return readTensorFromCSV(context, datasetFileName, false);
    }

    public static Tensor getNextSubtensor(DaalContext context, Tensor inputTensor, int startPos, int nElements) {
        long[] dims = inputTensor.getDimensions();
        dims[0] = nElements;

        int nDims = dims.length;
        int bufLength = nElements;

        for(int i = 1; i < nDims; i++) {
            bufLength *= dims[i];
        }

        FloatBuffer dataFloatSubtensor = FloatBuffer.allocate(bufLength);

        long fDims[] = {};
        inputTensor.getSubtensor(fDims, (long)startPos, (long)nElements, dataFloatSubtensor);

        float[] subtensorData = new float[bufLength];
        dataFloatSubtensor.get(subtensorData);

        inputTensor.releaseSubtensor(fDims, (long)startPos, (long)nElements, dataFloatSubtensor);

        return new HomogenTensor(context, dims, subtensorData);
    }
}
