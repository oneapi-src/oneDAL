/* file: NaiveBayesDenseStep2TrainingReducerAndPrediction.java */
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

package DAAL;

import java.io.*;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.fs.FileSystem;

import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.training.PartialResultId;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.multinomial_naive_bayes.*;
import com.intel.daal.algorithms.multinomial_naive_bayes.training.*;
import com.intel.daal.data_management.data.*;

import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.multinomial_naive_bayes.prediction.*;
import com.intel.daal.services.*;

public class NaiveBayesDenseStep2TrainingReducerAndPrediction extends
    Reducer<IntWritable, WriteableData, IntWritable, WriteableData> {

    private static final long nClasses = 20;

    private static final int nDataFeatures   = 20;
    private static final int nLabelsFeatures = 1;
    private static final int nVectors        = 2000;

    @Override
    public void reduce(IntWritable key, Iterable<WriteableData> values, Context context)
    throws IOException, InterruptedException {

        DaalContext daalContext = new DaalContext();

        /* Create an algorithm to train the Naive Bayes algorithm on the master node */
        TrainingDistributedStep2Master naiveBayesMaster = new TrainingDistributedStep2Master(daalContext, Double.class,
                                                                                             TrainingMethod.defaultDense,
                                                                                             nClasses);
        for (WriteableData value : values) {
            TrainingPartialResult pr = (TrainingPartialResult)value.getObject(daalContext);
            naiveBayesMaster.input.add(TrainingDistributedInputId.partialModels, pr);
        }
        /* Train the Naive Bayes algorithm on the master node */
        naiveBayesMaster.compute();

        /* Finalize computations and retrieve the results */
        TrainingResult res = naiveBayesMaster.finalizeCompute();

        Model model = res.get(TrainingResultId.model);

        /* Test the Naive Bayes model */
        prediction(daalContext, model, context);

        daalContext.dispose();
    }

    public void prediction(DaalContext daalContext, Model model, Context context) throws IOException, InterruptedException {
        /* Read a data set */
        String dataFilePath = "/Hadoop/NaiveBayesDense/data/NaiveBayesDense_test.csv";
        String labelsFilePath = "/Hadoop/NaiveBayesDense/data/NaiveBayesDense_test_labels.csv";

        double[] data = new double[nDataFeatures * nVectors];
        double[] labels = new double[nVectors];

        readData(dataFilePath, nDataFeatures, nVectors, data);
        readData(labelsFilePath, nLabelsFeatures, nVectors, labels);

        HomogenNumericTable ntData = new HomogenNumericTable(daalContext, data, nDataFeatures, nVectors);
        HomogenNumericTable expectedLabels = new HomogenNumericTable(daalContext, labels, nLabelsFeatures, nVectors);

        /* Create an algorithm to train the Naive Bayes algorithm on local nodes */
        PredictionBatch algorithm = new PredictionBatch(daalContext, Double.class, PredictionMethod.defaultDense, nClasses);

        algorithm.input.set(NumericTableInputId.data, ntData);
        algorithm.input.set(ModelInputId.model, model);

        /* Test the Naive Bayes algorithm on local nodes */
        PredictionResult res = algorithm.compute();

        /* Retrieve the results */
        HomogenNumericTable predictedlabels = (HomogenNumericTable)res.get(PredictionResultId.prediction);

        context.write(new IntWritable(0), new WriteableData(predictedlabels));
        context.write(new IntWritable(1), new WriteableData(expectedLabels));
    }

    private static void readData(String dataset, int nFeatures, int nVectors, double[] data) {
        System.out.println("readData " + dataset);
        try {
            Path pt = new Path(dataset);
            FileSystem fs = FileSystem.get(new Configuration());
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fs.open(pt)));

            int nLine = 0;
            for (String line; ((line = bufferedReader.readLine()) != null) && (nLine < nVectors); nLine++) {
                String[] elements = line.split(",");
                for (int j = 0; j < nFeatures; j++) {
                    data[nLine * nFeatures + j] = Double.parseDouble(elements[j]);
                }
            }
            bufferedReader.close();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        catch (NumberFormatException e) {
            e.printStackTrace();
        }
    }
}
