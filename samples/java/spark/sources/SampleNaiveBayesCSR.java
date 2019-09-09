/* file: SampleNaiveBayesCSR.java */
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
 //     Java sample of Naive Bayes classification.
 //
 //     The program trains the Naive Bayes model on a supplied training data set
 //     and then performs classification of previously unseen data.
 ////////////////////////////////////////////////////////////////////////////////
 */

package DAAL;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.nio.IntBuffer;
import java.io.*;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import scala.Tuple2;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

public class SampleNaiveBayesCSR {
    public static void main(String[] args) throws IOException {
        DaalContext context = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Spark Naive Bayes"));
        StringDataSource templateDataSource = new StringDataSource( context, "" );

        String trainDataFilesPath       = "/Spark/NaiveBayesCSR/data/NaiveBayesCSR_train_?.csv";
        String trainDataLabelsFilesPath = "/Spark/NaiveBayesCSR/data/NaiveBayesCSR_train_labels_?.csv";
        String testDataFilesPath        = "/Spark/NaiveBayesCSR/data/NaiveBayesCSR_test_1.csv";
        String testDataLabelsFilesPath  = "/Spark/NaiveBayesCSR/data/NaiveBayesCSR_test_labels_1.csv";

        /* Read the training data and labels from a specified path */
        JavaRDD<Tuple2<CSRNumericTable, HomogenNumericTable>> trainDataAndLabelsRDD =
            DistributedHDFSDataSet.getMergedCSRDataAndLabelsRDD(trainDataFilesPath, trainDataLabelsFilesPath, sc, templateDataSource);

        /* Read the test data and labels from a specified path */
        JavaRDD<Tuple2<CSRNumericTable, HomogenNumericTable>> testDataAndLabelsRDD =
            DistributedHDFSDataSet.getMergedCSRDataAndLabelsRDD(testDataFilesPath, testDataLabelsFilesPath, sc, templateDataSource);

        /* Compute the results of the Naive Bayes algorithm for dataRDD */
        SparkNaiveBayesCSR.NaiveBayesResult result = SparkNaiveBayesCSR.runNaiveBayes(context, trainDataAndLabelsRDD, testDataAndLabelsRDD);

        /* Print the results */
        HomogenNumericTable expected = null;
        List<Tuple2<CSRNumericTable, HomogenNumericTable>> parts_List = testDataAndLabelsRDD.collect();
        for (Tuple2<CSRNumericTable, HomogenNumericTable> value : parts_List) {
            expected = value._2;
            expected.unpack( context );
        }
        HomogenNumericTable predicted = result.prediction;

        printClassificationResult(expected, predicted, "Ground truth", "Classification results",
                                  "NaiveBayes classification results (first 20 observations):", 20);
        context.dispose();
        sc.stop();
    }

    public static void printClassificationResult(NumericTable groundTruth, NumericTable classificationResults,
                                                 String header1, String header2, String message, int nMaxRows) {
        int nCols = (int) groundTruth.getNumberOfColumns();
        int nRows = Math.min((int) groundTruth.getNumberOfRows(), nMaxRows);

        IntBuffer dataGroundTruth = IntBuffer.allocate(nCols * nRows);
        dataGroundTruth = groundTruth.getBlockOfRows(0, nRows, dataGroundTruth);

        IntBuffer dataClassificationResults = IntBuffer.allocate(nCols * nRows);
        dataClassificationResults = classificationResults.getBlockOfRows(0, nRows, dataClassificationResults);

        System.out.println(message);
        System.out.println(header1 + "\t" + header2);
        for(int i = 0; i < nRows; i++) {
            for(int j = 0; j < 1; j++) {
                System.out.format("%+d\t\t%+d\n", dataGroundTruth.get(i * nCols + j), dataClassificationResults.get(i * nCols + j));
            }
        }
    }
}
