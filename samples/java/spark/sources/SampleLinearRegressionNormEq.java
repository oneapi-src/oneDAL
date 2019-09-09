/* file: SampleLinearRegressionNormEq.java */
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
 //     Java sample of multiple linear regression.
 //
 //     The program trains the multiple linear regression model on a training
 //     data set with the normal equations method and computes regression for
 //     the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

package DAAL;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.nio.IntBuffer;
import java.nio.DoubleBuffer;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import scala.Tuple2;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

public class SampleLinearRegressionNormEq {
    public static void main(String[] args) {
        DaalContext context = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        StringDataSource templateDataSource = new StringDataSource( context, "" );
        JavaSparkContext sc                 = new JavaSparkContext(new SparkConf().setAppName("Spark Linear Regression"));

        String trainDataFilesPath       = "/Spark/LinearRegressionNormEq/data/LinearRegressionNormEq_train_?.csv";
        String trainDataLabelsFilesPath = "/Spark/LinearRegressionNormEq/data/LinearRegressionNormEq_train_labels_?.csv";
        String testDataFilesPath        = "/Spark/LinearRegressionNormEq/data/LinearRegressionNormEq_test_1.csv";
        String testDataLabelsFilesPath  = "/Spark/LinearRegressionNormEq/data/LinearRegressionNormEq_test_labels_1.csv";

        /* Read the training data and labels from a specified path */
        JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> trainDataAndLabelsRDD =
            DistributedHDFSDataSet.getMergedDataAndLabelsRDD(trainDataFilesPath, trainDataLabelsFilesPath, sc, templateDataSource);

        /* Read the test data and labels from a specified path */
        JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> testDataAndLabelsRDD =
            DistributedHDFSDataSet.getMergedDataAndLabelsRDD(testDataFilesPath, testDataLabelsFilesPath, sc, templateDataSource);

        /* Compute linear regression for dataRDD */
        SparkLinearRegressionNormEq.LinearRegressionResult result = SparkLinearRegressionNormEq.runLinearRegression(context, trainDataAndLabelsRDD,
                                                                                                                    testDataAndLabelsRDD);
        /* Print the results */
        HomogenNumericTable expected = null;

        List<Tuple2<HomogenNumericTable, HomogenNumericTable>> parts_List = testDataAndLabelsRDD.collect();
        for (Tuple2<HomogenNumericTable, HomogenNumericTable> value : parts_List) {
            expected = value._2;
            expected.unpack(context);
        }
        HomogenNumericTable predicted = result.prediction;
        HomogenNumericTable beta      = result.beta;

        printNumericTable("Coefficients:", beta);
        printNumericTable("First 10 rows of results (obtained): ", predicted, 10);
        printNumericTable("First 10 rows of results (expected): ", expected, 10);

        context.dispose();
        sc.stop();
    }

    public static void printNumericTable(String header, NumericTable nt, long nPrintedRows, long nPrintedCols) {
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

    public static void printNumericTable(String header, NumericTable nt) {
        printNumericTable(header, nt, nt.getNumberOfRows());
    }

    public static void printNumericTable(String header, NumericTable nt, long nRows) {
        printNumericTable(header, nt, nRows, nt.getNumberOfColumns());
    }
}
