/* file: SparkLinearRegressionNormEq.java */
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
//      Java sample of multiple linear regression in the distributed processing
//      mode.
//
//      The program trains the multiple linear regression model on a training
//      data set with the normal equations method and computes regression for
//      the test data.
////////////////////////////////////////////////////////////////////////////////
*/

package DAAL;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import com.intel.daal.algorithms.linear_regression.*;
import com.intel.daal.algorithms.linear_regression.training.*;
import com.intel.daal.algorithms.linear_regression.prediction.*;

import scala.Tuple2;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class SparkLinearRegressionNormEq {
    /* Class containing the algorithm results */
    static class LinearRegressionResult {
        public HomogenNumericTable prediction;
        public HomogenNumericTable beta;
    }

    public static LinearRegressionResult runLinearRegression(DaalContext context,
                                                             JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> trainDataRDD,
                                                             JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> testDataRDD) {
        JavaRDD<PartialResult> partsRDD = trainLocal(context, trainDataRDD);
        Model model = trainMaster(context, partsRDD);

        HomogenNumericTable prediction = testModel(context, model, testDataRDD);

        LinearRegressionResult result = new LinearRegressionResult();
        result.prediction = prediction;
        result.beta = (HomogenNumericTable)model.getBeta();
        return result;
    }

    private static JavaRDD<PartialResult> trainLocal(DaalContext context, JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> trainDataRDD) {
        return trainDataRDD.map(new Function<Tuple2<HomogenNumericTable, HomogenNumericTable>, PartialResult>() {
            public PartialResult call(Tuple2<HomogenNumericTable, HomogenNumericTable> tup) {
                DaalContext localContext = new DaalContext();

                /* Create an algorithm object to train the multiple linear regression model with the normal equations method */
                TrainingDistributedStep1Local linearRegressionTraining = new TrainingDistributedStep1Local(localContext, Double.class,
                                                                                                           TrainingMethod.normEqDense);
                /* Set the input data on local nodes */
                tup._1.unpack(localContext);
                tup._2.unpack(localContext);
                linearRegressionTraining.input.set(TrainingInputId.data, tup._1);
                linearRegressionTraining.input.set(TrainingInputId.dependentVariable, tup._2);

                /* Build a partial multiple linear regression model */
                PartialResult pres = linearRegressionTraining.compute();
                pres.pack();

                localContext.dispose();
                return pres;
            }
        });
    }

    private static Model trainMaster(DaalContext context, JavaRDD<PartialResult> partRes) {

        /* Create an algorithm object to train the multiple linear regression model with the normal equations method */
        TrainingDistributedStep2Master linearRegressionTraining = new TrainingDistributedStep2Master(context, Double.class,
                                                                                                     TrainingMethod.normEqDense);
        List<PartialResult> parts_List = partRes.collect();

        /* Add partial results computed on local nodes to the algorithm on the master node */
        for (PartialResult value : parts_List) {
            value.unpack(context);
            linearRegressionTraining.input.add(MasterInputId.partialModels, value);
        }

        /* Build and retrieve the final multiple linear regression model */
        linearRegressionTraining.compute();

        /* Finalize computations and retrieve the results */
        TrainingResult trainingResult = linearRegressionTraining.finalizeCompute();

        return trainingResult.get(TrainingResultId.model);
    }

    private static HomogenNumericTable testModel(DaalContext context,
                                                 final Model model,
                                                 JavaRDD<Tuple2<HomogenNumericTable, HomogenNumericTable>> testData) {

        /* Create algorithm objects to predict values of multiple linear regression with the default method */
        PredictionBatch linearRegressionPredict = new PredictionBatch(context, Double.class, PredictionMethod.defaultDense);

        /* Pass the test data to the algorithm */
        List<Tuple2<HomogenNumericTable, HomogenNumericTable>> parts_List = testData.collect();
        for (Tuple2<HomogenNumericTable, HomogenNumericTable> value : parts_List) {
            value._1.unpack(context);
            linearRegressionPredict.input.set(PredictionInputId.data, value._1);
        }

        linearRegressionPredict.input.set(PredictionInputId.model, model);

        /* Compute and retrieve the prediction results */
        PredictionResult predictionResult = linearRegressionPredict.compute();

        return (HomogenNumericTable)predictionResult.get(PredictionResultId.prediction);
    }
}
