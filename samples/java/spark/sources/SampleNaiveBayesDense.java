/* file: SampleNaiveBayesDense.java */
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

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import scala.Tuple2;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

public class SampleNaiveBayesDense {
    public static void main(String[] args) {
        DaalContext context = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Spark Naive Bayes"));
        StringDataSource templateDataSource = new StringDataSource( context, "" );

        String trainDataFilesPath       = "/Spark/NaiveBayesDense/data/NaiveBayesDense_train_?.csv";
        String trainDataLabelsFilesPath = "/Spark/NaiveBayesDense/data/NaiveBayesDense_train_labels_?.csv";
        String testDataFilesPath        = "/Spark/NaiveBayesDense/data/NaiveBayesDense_test_1.csv";
        String testDataLabelsFilesPath  = "/Spark/NaiveBayesDense/data/NaiveBayesDense_test_labels_1.csv";

        /* Read the training data and labels from a specified path */
        JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> trainDataAndLabelsRDD =
            DistributedHDFSDataSet.getMergedDataAndLabelsRDD(trainDataFilesPath, trainDataLabelsFilesPath, sc, templateDataSource);

        /* Read the test data and labels from a specified path */
        JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> testDataAndLabelsRDD =
            DistributedHDFSDataSet.getMergedDataAndLabelsRDD(testDataFilesPath, testDataLabelsFilesPath, sc, templateDataSource);

        /* Compute the results of the Naive Bayes algorithm for dataRDD */
        SparkNaiveBayesDense.NaiveBayesResult result = SparkNaiveBayesDense.runNaiveBayes(context, trainDataAndLabelsRDD, testDataAndLabelsRDD);

        /* Print the results */
        HomogenNumericTable expected = null;
        List<Tuple2<HomogenNumericTable, HomogenNumericTable>> parts_List = testDataAndLabelsRDD.collect();
        for (Tuple2<HomogenNumericTable, HomogenNumericTable> value : parts_List) {
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
