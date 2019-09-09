/* file: SampleLowOrderMomentsCSR.java */
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

/*
 //  Content:
 //     Java sample of computing low order moments.
 //
 //     Input matrix is stored in the compressed sparse row (CSR) format.
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

import java.io.*;
import scala.Tuple2;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

public class SampleLowOrderMomentsCSR {
    public static void main(String[] args) throws IOException {
        DaalContext context = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Spark low_order_moments(sparse)"));

        /* Read from the distributed HDFS data set at a specified path */
        StringDataSource templateDataSource = new StringDataSource( context, "" );
        DistributedHDFSDataSet dd = new DistributedHDFSDataSet( "/Spark/LowOrderMomentsCSR/data/", templateDataSource );
        JavaRDD<CSRNumericTable> dataRDD = dd.getCSRAsRDD(sc);

        /* Compute low order moments for dataRDD */
        SparkLowOrderMomentsCSR.MomentsResult result = SparkLowOrderMomentsCSR.runMoments(context, dataRDD);

        /* Print the results */
        HomogenNumericTable minimum              = result.minimum;
        HomogenNumericTable maximum              = result.maximum;
        HomogenNumericTable sum                  = result.sum;
        HomogenNumericTable sumSquares           = result.sumSquares;
        HomogenNumericTable sumSquaresCentered   = result.sumSquaresCentered;
        HomogenNumericTable mean                 = result.mean;
        HomogenNumericTable secondOrderRawMoment = result.secondOrderRawMoment;
        HomogenNumericTable variance             = result.variance;
        HomogenNumericTable standardDeviation    = result.standardDeviation;
        HomogenNumericTable variation            = result.variation;

        System.out.println("Low order moments:");
        printNumericTable("Min:", minimum);
        printNumericTable("Max:", maximum);
        printNumericTable("Sum:", sum);
        printNumericTable("SumSquares:", sumSquares);
        printNumericTable("SumSquaredDiffFromMean:", sumSquaresCentered);
        printNumericTable("Mean:", mean);
        printNumericTable("SecondOrderRawMoment:", secondOrderRawMoment);
        printNumericTable("Variance:", variance);
        printNumericTable("StandartDeviation:", standardDeviation);
        printNumericTable("Variation:", variation);

        context.dispose();
        sc.stop();
    }

    private static void printNumericTable(String header, HomogenNumericTable nt) {
        long nRows = nt.getNumberOfRows();
        long nCols = nt.getNumberOfColumns();
        double[] result = nt.getDoubleArray();

        int resultIndex = 0;
        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n");
        for (long i = 0; i < nRows; i++) {
            for (long j = 0; j < nCols; j++) {
                String tmp = String.format("%-6.3f   ", result[resultIndex++]);
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }
}
