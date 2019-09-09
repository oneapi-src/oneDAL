/* file: DistributedHDFSDataSet.java */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

package DAAL;

import com.intel.daal.services.Disposable;
import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

import java.io.*;
import java.util.*;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import scala.Tuple2;

import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.NumericTable;

/**
 * @brief Model is the base class for classes that represent models, such as
 * a linear regression or support vector machine (SVM) classifier.
 */
public class DistributedHDFSDataSet extends DistributedDataSet implements Serializable {

    /**
     * @brief Default constructor
     */
    public DistributedHDFSDataSet( String filename, StringDataSource dds ) {
        _filename = filename;
    }

    public DistributedHDFSDataSet( String filename, String labelsfilename, StringDataSource dds ) {
        _filename = filename;
        _labelsfilename = labelsfilename;
    }
    protected String _filename;
    protected String _labelsfilename;

    public JavaPairRDD<Integer, HomogenNumericTable> getAsPairRDDPartitioned( JavaSparkContext sc, int minPartitions, final long maxRowsPerTable ) {
        JavaRDD<String> rawData = sc.textFile(_filename, minPartitions);
        JavaPairRDD<String, Long> dataWithId = rawData.zipWithIndex();

        JavaPairRDD<Integer, HomogenNumericTable> data = dataWithId.mapPartitionsToPair(
        new PairFlatMapFunction<Iterator<Tuple2<String, Long>>, Integer, HomogenNumericTable>() {
            public Iterator<Tuple2<Integer, HomogenNumericTable>> call(Iterator<Tuple2<String, Long>> it) {

                DaalContext context = new DaalContext();
                long maxRows = maxRowsPerTable;
                long curRow  = 0;
                ArrayList<Tuple2<Integer, HomogenNumericTable>> tables = new ArrayList<Tuple2<Integer, HomogenNumericTable>>();

                StringDataSource dataSource = new StringDataSource( context, "" );

                while (it.hasNext()) {

                    dataSource.setData( it.next()._1 );
                    dataSource.loadDataBlock( 1, curRow, maxRows );

                    curRow++;

                    if( curRow == maxRows || !(it.hasNext()) ) {
                        HomogenNumericTable table = (HomogenNumericTable)dataSource.getNumericTable();
                        table.setNumberOfRows(curRow);
                        table.pack();

                        Tuple2<Integer, HomogenNumericTable> tuple = new Tuple2<Integer, HomogenNumericTable>(0, table);
                        tables.add(tuple);

                        dataSource = new StringDataSource( context, "" );

                        curRow = 0;
                    }
                }

                context.dispose();

                return tables.iterator();
            }
        } );

        return data;
    }

    public JavaPairRDD<Integer, HomogenNumericTable> getAsPairRDD( JavaSparkContext sc ) {

        JavaPairRDD<Tuple2<String, String>, Long> dataWithId = sc.wholeTextFiles(_filename).zipWithIndex();

        JavaPairRDD<Integer, HomogenNumericTable> data = dataWithId.mapToPair(
        new PairFunction<Tuple2<Tuple2<String, String>, Long>, Integer, HomogenNumericTable>() {
            public Tuple2<Integer, HomogenNumericTable> call(Tuple2<Tuple2<String, String>, Long> tup) {
                DaalContext context = new DaalContext();

                HomogenNumericTable dataTable = createHomogenTable(context, tup._1._2);
                dataTable.pack();

                context.dispose();

                return new Tuple2<Integer, HomogenNumericTable>(tup._2.intValue(), dataTable);
            }
        });

        return data;
    }

    public JavaRDD<HomogenNumericTable> getAsRDD( JavaSparkContext sc ) {

        JavaPairRDD<String, String> dataWithId = sc.wholeTextFiles(_filename);

        JavaRDD<HomogenNumericTable> data = dataWithId.map(new Function<Tuple2<String, String>, HomogenNumericTable>() {
            public HomogenNumericTable call(Tuple2<String, String> tup) {
                DaalContext context = new DaalContext();

                HomogenNumericTable dataTable = createHomogenTable(context, tup._2);
                dataTable.pack();

                context.dispose();

                return dataTable;
            }
        });

        return data;
    }

    public JavaPairRDD<Integer, HomogenNumericTable> getAsPairRDDWithIndex( JavaSparkContext sc ) {
        JavaPairRDD<Tuple2<String, String>, Long> dataWithId = sc.wholeTextFiles(_filename).zipWithIndex();

        JavaPairRDD<Integer, HomogenNumericTable> data = dataWithId.mapToPair(
        new PairFunction<Tuple2<Tuple2<String, String>, Long>, Integer, HomogenNumericTable>() {
            public Tuple2<Integer, HomogenNumericTable> call(Tuple2<Tuple2<String, String>, Long> tup) {
                DaalContext context = new DaalContext();

                HomogenNumericTable dataTable = createHomogenTable(context, tup._1._2);
                dataTable.pack();

                context.dispose();

                String fileName = tup._1._1;
                String[] tokens = fileName.split("[_\\.]");
                return new Tuple2<Integer, HomogenNumericTable>(Integer.parseInt(tokens[tokens.length - 2]) - 1, dataTable);
            }
        } );

        return data;
    }

    public JavaPairRDD<Integer, CSRNumericTable> getCSRAsPairRDD( JavaSparkContext sc ) throws IOException {

        JavaPairRDD<Tuple2<String, String>, Long> dataWithId = sc.wholeTextFiles(_filename).zipWithIndex();

        JavaPairRDD<Integer, CSRNumericTable> data = dataWithId.mapToPair(
        new PairFunction<Tuple2<Tuple2<String, String>, Long>, Integer, CSRNumericTable>() {
            public Tuple2<Integer, CSRNumericTable> call(Tuple2<Tuple2<String, String>, Long> tup) throws IOException {
                DaalContext context = new DaalContext();

                CSRNumericTable dataTable = createSparseTable(context, tup._1._2);
                dataTable.pack();

                context.dispose();

                return new Tuple2<Integer, CSRNumericTable>(tup._2.intValue(), dataTable);
            }
        } );

        return data;
    }

    public JavaRDD<CSRNumericTable> getCSRAsRDD( JavaSparkContext sc ) throws IOException {
        JavaPairRDD<String, String> dataWithId = sc.wholeTextFiles(_filename);

        JavaRDD<CSRNumericTable> data = dataWithId.map(new Function<Tuple2<String, String>, CSRNumericTable>() {
            public CSRNumericTable call(Tuple2<String, String> tup) throws IOException {
                DaalContext context = new DaalContext();

                CSRNumericTable dataTable = createSparseTable(context, tup._2);
                dataTable.pack();

                context.dispose();

                return dataTable;
            }
        });

        return data;
    }

    public JavaPairRDD<Integer, NumericTable> getCSRAsPairRDDWithIndex( JavaSparkContext sc ) throws IOException {

        JavaPairRDD<Tuple2<String, String>, Long> dataWithId = sc.wholeTextFiles(_filename).zipWithIndex();

        JavaPairRDD<Integer, NumericTable> data = dataWithId.mapToPair(
        new PairFunction<Tuple2<Tuple2<String, String>, Long>, Integer, NumericTable>() {
            public Tuple2<Integer, NumericTable> call(Tuple2<Tuple2<String, String>, Long> tup) throws IOException {

                DaalContext context = new DaalContext();

                String data = tup._1._2;

                CSRNumericTable dataTable = createSparseTable(context, data);
                dataTable.pack();

                context.dispose();

                String fileName = tup._1._1;
                String[] tokens = fileName.split("[_\\.]");
                return new Tuple2<Integer, NumericTable>(Integer.parseInt(tokens[tokens.length - 2]) - 1, (NumericTable)dataTable);
            }
        } );

        return data;
    }

    public static JavaPairRDD<Integer, Tuple2<HomogenNumericTable, HomogenNumericTable>> getMergedDataAndLabelsPairRDD( String trainDatafilesPath,
                                                                                                                        String trainDataLabelsfilesPath,
                                                                                                                        JavaSparkContext sc,
                                                                                                                        StringDataSource tempDataSource ) {

        DistributedHDFSDataSet ddTrain  = new DistributedHDFSDataSet(trainDatafilesPath, tempDataSource );
        DistributedHDFSDataSet ddLabels = new DistributedHDFSDataSet(trainDataLabelsfilesPath, tempDataSource );

        JavaPairRDD<Integer, HomogenNumericTable> dataRDD   = ddTrain.getAsPairRDDWithIndex(sc);
        JavaPairRDD<Integer, HomogenNumericTable> labelsRDD = ddLabels.getAsPairRDDWithIndex(sc);

        JavaPairRDD<Integer, Tuple2<Iterable<HomogenNumericTable>, Iterable<HomogenNumericTable>>> dataAndLablesRDD = dataRDD.cogroup(labelsRDD);

        JavaPairRDD<Integer, Tuple2<HomogenNumericTable, HomogenNumericTable>> mergedDataAndLabelsRDD = dataAndLablesRDD.mapToPair(
        new PairFunction<Tuple2<Integer, Tuple2<Iterable<HomogenNumericTable>, Iterable<HomogenNumericTable>>>, Integer, Tuple2<HomogenNumericTable,
                                                                                                                           HomogenNumericTable>>() {

            public Tuple2<Integer, Tuple2<HomogenNumericTable, HomogenNumericTable>>
            call(Tuple2<Integer, Tuple2<Iterable<HomogenNumericTable>, Iterable<HomogenNumericTable>>> tup) {

                HomogenNumericTable dataNT = tup._2._1.iterator().next();
                HomogenNumericTable labelsNT = tup._2._2.iterator().next();

                return new Tuple2<Integer, Tuple2<HomogenNumericTable, HomogenNumericTable>>(tup._1, new Tuple2<HomogenNumericTable,
                                                                                             HomogenNumericTable>(dataNT, labelsNT));
            }
        }).cache();

        mergedDataAndLabelsRDD.count();
        return mergedDataAndLabelsRDD;
    }

    public static JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> getMergedDataAndLabelsRDD( String trainDatafilesPath,
                                                                                                       String trainDataLabelsfilesPath,
                                                                                                       JavaSparkContext sc,
                                                                                                       StringDataSource tempDataSource ) {
        DistributedHDFSDataSet ddTrain  = new DistributedHDFSDataSet(trainDatafilesPath, tempDataSource);
        DistributedHDFSDataSet ddLabels = new DistributedHDFSDataSet(trainDataLabelsfilesPath, tempDataSource);

        JavaPairRDD<Integer, HomogenNumericTable> dataRDD   = ddTrain.getAsPairRDDWithIndex(sc);
        JavaPairRDD<Integer, HomogenNumericTable> labelsRDD = ddLabels.getAsPairRDDWithIndex(sc);

        JavaPairRDD<Integer, Tuple2<Iterable<HomogenNumericTable>, Iterable<HomogenNumericTable>>> dataAndLablesRDD = dataRDD.cogroup(labelsRDD);

        JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> mergedDataAndLabelsRDD = dataAndLablesRDD.map(
        new Function<Tuple2<Integer, Tuple2<Iterable<HomogenNumericTable>, Iterable<HomogenNumericTable>>>,
            Tuple2<HomogenNumericTable, HomogenNumericTable>>() {

                public Tuple2<HomogenNumericTable, HomogenNumericTable> call(Tuple2<Integer, Tuple2<Iterable<HomogenNumericTable>,
                    Iterable<HomogenNumericTable>>> tup) {

                    HomogenNumericTable dataNT = tup._2._1.iterator().next();
                    HomogenNumericTable labelsNT = tup._2._2.iterator().next();

                    return new Tuple2<HomogenNumericTable, HomogenNumericTable>(dataNT, labelsNT);
                }
        }).cache();

        mergedDataAndLabelsRDD.count();
        return mergedDataAndLabelsRDD;
    }


    public static JavaPairRDD<Integer, Tuple2<CSRNumericTable, HomogenNumericTable>> getMergedCSRDataAndLabelsPairRDD(String trainDatafilesPath,
             String trainDataLabelsfilesPath, JavaSparkContext sc, StringDataSource tempDataSource) throws IOException {

        DistributedHDFSDataSet ddTrain  = new DistributedHDFSDataSet(trainDatafilesPath, tempDataSource );
        DistributedHDFSDataSet ddLabels = new DistributedHDFSDataSet(trainDataLabelsfilesPath, tempDataSource );

        JavaPairRDD<Integer, CSRNumericTable> dataRDD   = ddTrain.getCSRAsPairRDD(sc);
        JavaPairRDD<Integer, HomogenNumericTable> labelsRDD = ddLabels.getAsPairRDDWithIndex(sc);

        JavaPairRDD<Integer, Tuple2<Iterable<CSRNumericTable>, Iterable<HomogenNumericTable>>> dataAndLablesRDD = dataRDD.cogroup(labelsRDD);

        JavaPairRDD<Integer, Tuple2<CSRNumericTable, HomogenNumericTable>> mergedCSRDataAndLabelsRDD = dataAndLablesRDD.mapToPair(
        new PairFunction<Tuple2<Integer, Tuple2<Iterable<CSRNumericTable>, Iterable<HomogenNumericTable>>>, Integer, Tuple2<CSRNumericTable,
                                                                                                                           HomogenNumericTable>>() {

            public Tuple2<Integer, Tuple2<CSRNumericTable, HomogenNumericTable>>
            call(Tuple2<Integer, Tuple2<Iterable<CSRNumericTable>, Iterable<HomogenNumericTable>>> tup) throws IOException {

                CSRNumericTable dataNT = tup._2._1.iterator().next();
                HomogenNumericTable labelsNT = tup._2._2.iterator().next();

                return new Tuple2<Integer, Tuple2<CSRNumericTable, HomogenNumericTable>>(tup._1, new Tuple2<CSRNumericTable,
                                                                                             HomogenNumericTable>(dataNT, labelsNT));
            }
        }).cache();

        mergedCSRDataAndLabelsRDD.count();
        return mergedCSRDataAndLabelsRDD;
    }

    public static JavaRDD<Tuple2<CSRNumericTable, HomogenNumericTable>> getMergedCSRDataAndLabelsRDD(String trainDatafilesPath,
             String trainDataLabelsfilesPath, JavaSparkContext sc, StringDataSource tempDataSource) throws IOException {

        DistributedHDFSDataSet ddTrain  = new DistributedHDFSDataSet(trainDatafilesPath, tempDataSource );
        DistributedHDFSDataSet ddLabels = new DistributedHDFSDataSet(trainDataLabelsfilesPath, tempDataSource );

        JavaPairRDD<Integer, CSRNumericTable> dataRDD   = ddTrain.getCSRAsPairRDD(sc);
        JavaPairRDD<Integer, HomogenNumericTable> labelsRDD = ddLabels.getAsPairRDDWithIndex(sc);

        JavaPairRDD<Integer, Tuple2<Iterable<CSRNumericTable>, Iterable<HomogenNumericTable>>> dataAndLablesRDD = dataRDD.cogroup(labelsRDD);

        JavaRDD<Tuple2<CSRNumericTable, HomogenNumericTable>> mergedCSRDataAndLabelsRDD = dataAndLablesRDD.map(
        new Function<Tuple2<Integer, Tuple2<Iterable<CSRNumericTable>, Iterable<HomogenNumericTable>>>, Tuple2<CSRNumericTable,
                                                                                                                           HomogenNumericTable>>() {

            public Tuple2<CSRNumericTable, HomogenNumericTable>
            call(Tuple2<Integer, Tuple2<Iterable<CSRNumericTable>, Iterable<HomogenNumericTable>>> tup) throws IOException {

                CSRNumericTable dataNT = tup._2._1.iterator().next();
                HomogenNumericTable labelsNT = tup._2._2.iterator().next();

                return new Tuple2<CSRNumericTable, HomogenNumericTable>(dataNT, labelsNT);
            }
        }).cache();

        mergedCSRDataAndLabelsRDD.count();
        return mergedCSRDataAndLabelsRDD;
    }

    public static CSRNumericTable createSparseTable(DaalContext context, String inputData) throws IOException {

        String[] elements = inputData.split("\n");

        String rowIndexLine = elements[0];
        String columnsLine = elements[1];
        String valuesLine = elements[2];

        int nVectors = getRowLength(rowIndexLine);
        long[] rowOffsets = new long[nVectors];

        readRow(rowIndexLine, 0, nVectors, rowOffsets);
        nVectors = nVectors - 1;

        int nCols = getRowLength(columnsLine);

        long[] colIndices = new long[nCols];
        readRow(columnsLine, 0, nCols, colIndices);

        int nNonZeros = getRowLength(valuesLine);

        double[] data = new double[nNonZeros];
        readRow(valuesLine, 0, nNonZeros, data);

        long maxCol = 0;
        for(int i = 0; i < nCols; i++) {
            if(colIndices[i] > maxCol) {
                maxCol = colIndices[i];
            }
        }
        int nFeatures = (int)maxCol;

        if (nCols != nNonZeros || nNonZeros != (rowOffsets[nVectors] - 1) || nFeatures == 0 || nVectors == 0) {
            throw new IOException("Unable to read input data");
        }

        return new CSRNumericTable(context, data, colIndices, rowOffsets, nFeatures, nVectors);
    }

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

    public static void readSparseData(String dataset, int nVectors, int nNonZeroValues,
                                      long[] rowOffsets, long[] colIndices, double[] data) {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(dataset));
            readRow(bufferedReader.readLine(), 0, nVectors + 1, rowOffsets);
            readRow(bufferedReader.readLine(), 0, nNonZeroValues, colIndices);
            readRow(bufferedReader.readLine(), 0, nNonZeroValues, data);
            bufferedReader.close();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        catch (NumberFormatException e) {
            e.printStackTrace();
        }
    }

    private static int getRowLength(String line) {
        String[] elements = line.split(",");
        return elements.length;
    }

    private static HomogenNumericTable createHomogenTable(DaalContext context, String data) {
        long nVectors = 0;
        for(int i = 0; i < data.length(); i++) {
            if(data.charAt(i) == '\n') {
                nVectors++;
            }
        }

        StringDataSource sdds = new StringDataSource( context, "" );
        sdds.setData( data );

        sdds.createDictionaryFromContext();
        sdds.allocateNumericTable();
        sdds.loadDataBlock( nVectors );

        HomogenNumericTable dataTable = (HomogenNumericTable)sdds.getNumericTable();

        return dataTable;
    }
}
