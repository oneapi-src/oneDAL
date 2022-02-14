/* file: SampleKmeansInitCSR.java */
/*******************************************************************************
* Copyright 2017 Intel Corporation
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
 //     Java sample of csr K-Means clustering
 ////////////////////////////////////////////////////////////////////////////////
 */

package DAAL;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import java.nio.DoubleBuffer;
import java.io.*;
import java.io.IOException;
import java.lang.ClassNotFoundException;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

public class SampleKmeansInitCSR {
    public static void main(String[] args) throws IOException, ClassNotFoundException, IllegalAccessException {
        DaalContext context = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Spark Kmeans"));

        /* Read from the distributed HDFS data set at a specified path */
        StringDataSource templateDataSource = new StringDataSource( context, "" );
        DistributedHDFSDataSet dd = new DistributedHDFSDataSet( "/Spark/KmeansInitCSR/data/", templateDataSource );
        JavaPairRDD<Integer, CSRNumericTable> dataRDD = dd.getCSRAsPairRDD(sc);

        /* Get initial centroids by plusPlusCSR method */
        /* Compute k-means for dataRDD */
        SparkKmeansInitCSR.KmeansInitResult initResult = SparkKmeansInitCSR.initKmeansPlusPlus(sc, context, dataRDD);
        SparkKmeansInitCSR.KmeansResult         result = SparkKmeansInitCSR.runKmeans(context, dataRDD, initResult.centroids);

        /* Print the results */
        printNumericTable("Initial centroids (plusPlusCSR method):", initResult.centroids, 20, 10);
        printNumericTable("Result centroids:", result.centroids, 20, 10);

        /* Get initial centroids by parallelPlusCSR method */
        /* Compute k-means for dataRDD */
        initResult = SparkKmeansInitCSR.initKmeansParallelPlus(sc, context, dataRDD);
        result     = SparkKmeansInitCSR.runKmeans(context, dataRDD, initResult.centroids);

        /* Print the results */
        printNumericTable("Initial centroids (parallelPlusCSR method):", initResult.centroids, 20, 10);
        printNumericTable("Result centroids:", result.centroids, 20, 10);

        context.dispose();
        sc.stop();
    }

    public static void printNumericTable(String header, NumericTable nt, long nPrintedRows, long nPrintedCols) throws IllegalAccessException {
        long nNtCols = nt.getNumberOfColumns();
        long nNtRows = nt.getNumberOfRows();
        long nRows = nNtRows;
        long nCols = nNtCols;

        if(nPrintedRows > 0) {
            nRows = Math.min(nNtRows, nPrintedRows);
        }

        DoubleBuffer result = DoubleBuffer.allocate((int)(nNtCols * nRows));
        result = nt.getBlockOfRows(0, nRows, result);

        if(nPrintedCols > 0) {
            nCols = Math.min(nNtCols, nPrintedCols);
        }

        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n");
        for (long i = 0; i < nRows; i++) {
            for (long j = 0; j < nCols; j++) {
                String tmp = String.format("%-6.3f   ", result.get((int)(i * nNtCols + j)));
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }
}
