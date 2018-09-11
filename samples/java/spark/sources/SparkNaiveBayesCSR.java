/* file: SparkNaiveBayesCSR.java */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
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
//      Java sample of Naive Bayes classification in the distributed processing
//      mode.
//
//      The program trains the Naive Bayes model on a supplied training data set
//      and then performs classification of previously unseen data.
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

import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.training.PartialResultId;

import com.intel.daal.algorithms.multinomial_naive_bayes.*;
import com.intel.daal.algorithms.multinomial_naive_bayes.training.*;
import com.intel.daal.algorithms.multinomial_naive_bayes.prediction.*;

import scala.Tuple2;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class SparkNaiveBayesCSR {
    /* Class containing the algorithm results */
    static class NaiveBayesResult {
        public HomogenNumericTable prediction;
    }

    private static final long nClasses          = 20;
    private static final int  nTestObservations = 2000;

    public static NaiveBayesResult runNaiveBayes(DaalContext context,
                                                 JavaRDD<Tuple2<CSRNumericTable, HomogenNumericTable>> trainDataRDD,
                                                 JavaRDD<Tuple2<CSRNumericTable, HomogenNumericTable>> testDataRDD) {
        JavaRDD<TrainingPartialResult> partsRDD = trainLocal(trainDataRDD);
        TrainingPartialResult finalPartRes = reducePartialResults(partsRDD);
        Model model = trainMaster(context, finalPartRes);

        HomogenNumericTable prediction = testModel(context, testDataRDD, model);

        NaiveBayesResult result = new NaiveBayesResult();
        result.prediction = prediction;
        return result;
    }

    private static JavaRDD<TrainingPartialResult> trainLocal(JavaRDD<Tuple2<CSRNumericTable, HomogenNumericTable>> trainDataRDD) {
        return trainDataRDD.map(new Function<Tuple2<CSRNumericTable, HomogenNumericTable>, TrainingPartialResult>() {
            public TrainingPartialResult call(Tuple2<CSRNumericTable, HomogenNumericTable> tup) {
                DaalContext context = new DaalContext();

                /* Create an algorithm to train the Naive Bayes model on local nodes */
                TrainingDistributedStep1Local algorithm = new TrainingDistributedStep1Local(context, Double.class, TrainingMethod.fastCSR,
                                                                                            nClasses);
                /* Set the input data on local nodes */
                tup._1.unpack(context);
                tup._2.unpack(context);
                algorithm.input.set(InputId.data, tup._1);
                algorithm.input.set(InputId.labels, tup._2);

                /* Train the Naive Bayes model on local nodes */
                TrainingPartialResult pres = algorithm.compute();
                pres.pack();

                context.dispose();
                return pres;
            }
        });
    }

    private static TrainingPartialResult reducePartialResults(JavaRDD<TrainingPartialResult> partsRDD) {
        return partsRDD.reduce(new Function2<TrainingPartialResult, TrainingPartialResult, TrainingPartialResult>() {
            public TrainingPartialResult call(TrainingPartialResult p1, TrainingPartialResult p2) {
                DaalContext context = new DaalContext();

                /* Create an algorithm to compute new partial result from two partial results */
                TrainingDistributedStep2Master algorithm = new TrainingDistributedStep2Master(context, Double.class,
                                                                                              TrainingMethod.fastCSR, nClasses);

                /* Set the partial results recieved from the local nodes */
                p1.unpack(context);
                p2.unpack(context);
                algorithm.input.add(TrainingDistributedInputId.partialModels, p1);
                algorithm.input.add(TrainingDistributedInputId.partialModels, p2);

                /* Compute a new partial result from two partial results */
                TrainingPartialResult pres = algorithm.compute();
                pres.pack();

                context.dispose();
                return pres;
            }
        });
    }

    private static Model trainMaster(DaalContext context, TrainingPartialResult partRes) {

        /* Create an algorithm to train the Naive Bayes model on the master node */
        TrainingDistributedStep2Master algorithm = new TrainingDistributedStep2Master(context, Double.class,
                                                                                      TrainingMethod.fastCSR, nClasses);

        /* Add partial results computed on local nodes to the algorithm on the master node */
        partRes.unpack(context);
        algorithm.input.add(TrainingDistributedInputId.partialModels, partRes);

        /* Train the Naive Bayes model on the master node */
        algorithm.compute();

        /* Finalize computations and retrieve the training results */
        TrainingResult trainingResult = algorithm.finalizeCompute();

        return trainingResult.get(TrainingResultId.model);
    }

    private static HomogenNumericTable testModel(DaalContext context, JavaRDD<Tuple2<CSRNumericTable, HomogenNumericTable>> testData,
                                                 final Model model) {

        /* Create algorithm objects to predict values of the Naive Bayes model with the fastCSR method */
        PredictionBatch algorithm = new PredictionBatch(context, Double.class, PredictionMethod.fastCSR, nClasses);

        /* Pass the test data to the algorithm */
        List<Tuple2<CSRNumericTable, HomogenNumericTable>> parts_List = testData.collect();
        for (Tuple2<CSRNumericTable, HomogenNumericTable> value : parts_List) {
            value._1.unpack(context);
            algorithm.input.set(NumericTableInputId.data, value._1);
        }
        algorithm.input.set(ModelInputId.model, model);

        /* Compute the prediction results */
        PredictionResult predictionResult = algorithm.compute();

        /* Retrieve the results */
        return (HomogenNumericTable)predictionResult.get(PredictionResultId.prediction);
    }
}
