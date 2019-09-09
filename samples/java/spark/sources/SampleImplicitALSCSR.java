/* file: SampleImplicitALSCSR.java */
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
 //     Java sample of the implicit alternating least squares (ALS) algorithm.
 //
 //     The program trains the implicit ALS trainedModel on a supplied training data
 //     set.
 ////////////////////////////////////////////////////////////////////////////////
 */

package DAAL;

import java.util.List;
import java.nio.DoubleBuffer;
import java.io.IOException;
import java.lang.ClassNotFoundException;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import scala.Tuple2;
import scala.Tuple3;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.algorithms.implicit_als.*;
import com.intel.daal.algorithms.implicit_als.training.*;
import com.intel.daal.algorithms.implicit_als.prediction.ratings.*;
import com.intel.daal.services.*;

public class SampleImplicitALSCSR {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        DaalContext context = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Spark Implicit ALS"));

        /* Read from the distributed HDFS data set at a specified path */
        StringDataSource templateDataSource = new StringDataSource(context, "");

        DistributedHDFSDataSet ddTrain = new DistributedHDFSDataSet("/Spark/ImplicitALSCSR/data/ImplicitALSCSRTrans_*",
                                                                    templateDataSource );

        JavaPairRDD<Integer, NumericTable> dataRDD = ddTrain.getCSRAsPairRDDWithIndex(sc);

        SparkImplicitALSCSR.TrainingResult trainedModel = SparkImplicitALSCSR.trainModel(sc, dataRDD);
        printTrainedModel(trainedModel);

        JavaRDD<Tuple3<Integer, Integer, RatingsResult>> predictedRatings =
            SparkImplicitALSCSR.testModel(trainedModel.usersFactors, trainedModel.itemsFactors);
        printPredictedRatings(predictedRatings);

        context.dispose();
        sc.stop();
    }

    public static void printTrainedModel(SparkImplicitALSCSR.TrainingResult trainedModel) {
        DaalContext context = new DaalContext();
        List<Tuple2<Integer, DistributedPartialResultStep4>> itemsFactorsList = trainedModel.itemsFactors.collect();
        for (Tuple2<Integer, DistributedPartialResultStep4> tup : itemsFactorsList) {
            tup._2.unpack(context);
            printNumericTable("Partial items factors " + tup._1 + " :",
                              tup._2.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1).getFactors());
            tup._2.pack();
        }

        List<Tuple2<Integer, DistributedPartialResultStep4>> usersFactorsList = trainedModel.usersFactors.collect();
        for (Tuple2<Integer, DistributedPartialResultStep4> tup : usersFactorsList) {
            tup._2.unpack(context);
            printNumericTable("Partial users factors " + tup._1 + " :",
                              tup._2.get(DistributedPartialResultStep4Id.outputOfStep4ForStep1).getFactors());
            tup._2.pack();
        }
        context.dispose();
    }

    public static void printPredictedRatings(JavaRDD<Tuple3<Integer, Integer, RatingsResult>> predictedRatings) {
        DaalContext context = new DaalContext();
        List<Tuple3<Integer, Integer, RatingsResult>> predictedRatingsList = predictedRatings.collect();
        for (Tuple3<Integer, Integer, RatingsResult> tup : predictedRatingsList) {
            tup._3().unpack(context);
            printNumericTable("Ratings [" + tup._1() + ", " + tup._2() + "]" , tup._3().get(RatingsResultId.prediction));
            tup._3().pack();
        }
        context.dispose();
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

    public static void printNumericTable(String header, NumericTable nt, long nRows) {
        printNumericTable(header, nt, nRows, nt.getNumberOfColumns());
    }

    public static void printNumericTable(String header, NumericTable nt) {
        printNumericTable(header, nt, nt.getNumberOfRows());
    }
}
