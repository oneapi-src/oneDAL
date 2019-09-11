/* file: SampleParquet.java */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
 //  Content:
 //     Java sample of using Parquet Data Frame
 ////////////////////////////////////////////////////////////////////////////////
 */

package DAAL;

import java.util.Date;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;

import java.nio.DoubleBuffer;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import scala.Tuple2;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

public class SampleParquet {
    private static final int rowsToPrint = 5;
    private static final int partitionsToPrint = 10;

    public static void main(String[] args) {
        DaalContext context = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Spark Parquet"));

        SparkSession sparkSession = SparkSession.builder().getOrCreate();

        Dataset<Row> df = sparkSession.read().parquet("/Spark/Parquet/data/Parquet");

        JavaRDD<HomogenNumericTable> dataRDD = convertDataFrame(df, 4096);

        List<HomogenNumericTable> dfList = dataRDD.collect();

        int parts = dfList.size();

        System.out.println(dataRDD.count() + " tables");

        for (int i = 0; i < parts && i < partitionsToPrint; i++) {
            HomogenNumericTable table = dfList.get(i);
            int nColumns = (int)table.getNumberOfColumns();
            int nRows = (int)table.getNumberOfRows();
            if (nRows > rowsToPrint) {
                nRows = rowsToPrint;
            }
            DoubleBuffer dataDouble = DoubleBuffer.allocate(nColumns * nRows);
            dataDouble = table.getBlockOfRows(0, nRows, dataDouble);
            printDoubleBuffer(dataDouble, nColumns, nRows, "Print homogen data table as double:");
            table.releaseBlockOfRows(0, nRows, dataDouble);
        }

        context.dispose();
        sc.stop();
    }

    private static JavaRDD<HomogenNumericTable> convertDataFrame(Dataset<Row> df, final long maxRowsPerTable) {

        JavaRDD<HomogenNumericTable> dataRDD = df.rdd().toJavaRDD().mapPartitions(new FlatMapFunction<Iterator<Row>, HomogenNumericTable>() {
            public Iterator<HomogenNumericTable> call(Iterator<Row> it) {
                DaalContext localContext = new DaalContext();
                long maxRows = maxRowsPerTable;
                long curRow  = 0;
                ArrayList<HomogenNumericTable> tables = new ArrayList<HomogenNumericTable>();
                ArrayList<Row> rows = new ArrayList<Row>();
                while (it.hasNext()) {
                    rows.add(it.next());
                    curRow++;
                    if( curRow == maxRows || !(it.hasNext()) ) {
                        long nCols = rows.get(0).length();
                        double[] data = new double[(int)(curRow * nCols)];
                        for (int i = 0; i < (int)curRow; i++) {
                            Row row = rows.get(i);
                            for (int j = 0; j < (int)nCols; j++) {
                                data[i * (int)nCols + j] = row.getDouble(j);
                            }
                        }
                        HomogenNumericTable table = new HomogenNumericTable(localContext, data, nCols, curRow);
                        table.pack();
                        tables.add(table);
                        rows.clear();
                        curRow = 0;
                    }
                }
                localContext.dispose();
                return tables.iterator();
            }
        });

        return dataRDD;
    }

    private static void printDoubleBuffer(DoubleBuffer buf, long nColumns, long nRows, String message) {
        int step = (int) nColumns;
        System.out.println(message);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nColumns; j++) {
                System.out.format("%6.3f   ", buf.get(i * step + j));
            }
            System.out.println("");
        }
        System.out.println("");
    }
}
