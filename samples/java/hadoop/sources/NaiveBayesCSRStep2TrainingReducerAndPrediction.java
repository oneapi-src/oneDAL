/* file: NaiveBayesCSRStep2TrainingReducerAndPrediction.java */
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
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.algorithms.multinomial_naive_bayes.*;
import com.intel.daal.algorithms.multinomial_naive_bayes.training.*;
import com.intel.daal.data_management.data.*;

import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.multinomial_naive_bayes.prediction.*;
import com.intel.daal.services.*;

public class NaiveBayesCSRStep2TrainingReducerAndPrediction extends
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
                                                                                             TrainingMethod.fastCSR,
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
        String dataFilePath = "/Hadoop/NaiveBayesCSR/data/NaiveBayesCSR_test.csv";
        String labelsFilePath = "/Hadoop/NaiveBayesCSR/data/NaiveBayesCSR_test_labels.csv";

        // double[] data = new double[nDataFeatures * nVectors];
        double[] labels = new double[nVectors];

        readData(labelsFilePath, nLabelsFeatures, nVectors, labels);

        CSRNumericTable ntData = createSparseTable(daalContext, dataFilePath);
        HomogenNumericTable expectedLabels = new HomogenNumericTable(daalContext, labels, nLabelsFeatures, nVectors);

        /* Create an algorithm to train the Naive Bayes algorithm on local nodes */
        PredictionBatch algorithm = new PredictionBatch(daalContext, Double.class, PredictionMethod.fastCSR, nClasses);

        algorithm.input.set(NumericTableInputId.data, ntData);
        algorithm.input.set(ModelInputId.model, model);

        /* Test the Naive Bayes algorithm on local nodes */
        PredictionResult res = algorithm.compute();

        /* Retrieve the results */
        HomogenNumericTable predictedlabels = (HomogenNumericTable)res.get(PredictionResultId.prediction);

        context.write(new IntWritable(0), new WriteableData(predictedlabels));
        context.write(new IntWritable(1), new WriteableData(expectedLabels));
    }

    public static CSRNumericTable createSparseTable(DaalContext daalContext, String dataset) throws IOException {
        Path pt = new Path(dataset);
        FileSystem fs = FileSystem.get(new Configuration());
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fs.open(pt)));

        String rowIndexLine = bufferedReader.readLine();
        int nVectors = getRowLength(rowIndexLine);
        long[] rowOffsets = new long[nVectors];

        readRow(rowIndexLine, 0, nVectors, rowOffsets);
        nVectors = nVectors - 1;

        String columnsLine = bufferedReader.readLine();
        int nCols = getRowLength(columnsLine);

        long[] colIndices = new long[nCols];
        readRow(columnsLine, 0, nCols, colIndices);

        String valuesLine = bufferedReader.readLine();
        int nNonZeros = getRowLength(valuesLine);

        double[] data = new double[nNonZeros];
        readRow(valuesLine, 0, nNonZeros, data);

        long maxCol = 0;
        for(int i = 0; i < nCols; i++) {
            if(colIndices[i] > maxCol) {
                maxCol = colIndices[i];
            }
        }
        int nFeatures = (int)maxCol;

        if (nCols != nNonZeros || nNonZeros != (rowOffsets[nVectors] - 1) || nFeatures == 0 || nVectors == 0) {
            throw new IOException("Unable to read input dataset");
        }

        return new CSRNumericTable(daalContext, data, colIndices, rowOffsets, nFeatures, nVectors);
    }

    private static int getRowLength(String line)  throws IOException {
        if (line == null) {
            throw new IOException("Unable to read input dataset");
        }

        String[] elements = line.split(",");
        return elements.length;
    }

    public static void readRow(String line, int offset, int nCols, double[] data) throws IOException {
        if (line == null) {
            throw new IOException("Unable to read input dataset");
        }

        String[] elements = line.split(",");
        for (int j = 0; j < nCols; j++) {
            data[offset + j] = Double.parseDouble(elements[j]);
        }
    }

    public static void readRow(String line, int offset, int nCols, long[] data) throws IOException {
        if (line == null) {
            throw new IOException("Unable to read input dataset");
        }

        String[] elements = line.split(",");
        for (int j = 0; j < nCols; j++) {
            data[offset + j] = Long.parseLong(elements[j]);
        }
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
