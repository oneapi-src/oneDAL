/* file: NaiveBayesDenseStep1TrainingMapper.java */
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

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.BufferedWriter;
import java.util.Arrays;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.conf.Configuration;

import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.training.PartialResultId;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.multinomial_naive_bayes.*;
import com.intel.daal.algorithms.multinomial_naive_bayes.training.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class NaiveBayesDenseStep1TrainingMapper extends Mapper<Object, Text, IntWritable, WriteableData> {

    private static final int nDataFeatures   = 20;
    private static final int nLabelsFeatures = 1;
    private static final int nVectorsInBlock = 2000;
    private static final long nClasses       = 20;

    /* Index is supposed to be a sequence number for the split */
    private int index = 0;
    private int totalTasks = 0;

    @Override
    public void setup(Context context) {
        index = context.getTaskAttemptID().getTaskID().getId();
        Configuration conf = context.getConfiguration();
        totalTasks = conf.getInt("mapred.map.tasks", 0);
    }

    @Override
    public void map(Object key, Text value,
                    Context context) throws IOException, InterruptedException {

        /* Read a data set */
        String dataFilePath = "/Hadoop/NaiveBayesDense/data/" + value + "_train.csv";
        String labelsFilePath = "/Hadoop/NaiveBayesDense/data/" + value + "_train_labels.csv";

        double[] data = new double[nDataFeatures * nVectorsInBlock];
        double[] labels = new double[nLabelsFeatures * nVectorsInBlock];

        readData(dataFilePath, nDataFeatures, nVectorsInBlock, data);
        readData(labelsFilePath, nLabelsFeatures, nVectorsInBlock, labels);

        DaalContext daalContext = new DaalContext();

        HomogenNumericTable ntData = new HomogenNumericTable(daalContext, data, nDataFeatures, nVectorsInBlock);
        HomogenNumericTable ntLabels = new HomogenNumericTable(daalContext, labels, nLabelsFeatures, nVectorsInBlock);

        /* Create an algorithm to train the Naive Bayes algorithm on local nodes */
        TrainingDistributedStep1Local naiveBayesLocal = new TrainingDistributedStep1Local(daalContext, Double.class,
                                                                                          TrainingMethod.defaultDense,
                                                                                          nClasses);
        naiveBayesLocal.input.set(InputId.data, ntData);
        naiveBayesLocal.input.set(InputId.labels, ntLabels);

        /* Train the Naive Bayes algorithm on local nodes */
        TrainingPartialResult pres = naiveBayesLocal.compute();

        /* Write the data prepended with a data set sequence number. Needed to know the position of the data set in the input data */
        context.write(new IntWritable(0), new WriteableData(index, pres));
        index += totalTasks;

        daalContext.dispose();
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

    public static void writeData(String fileName, double[] data, int nF, int nV) {
        try {
            Path pt = new Path(fileName);
            FileSystem fs = FileSystem.get(new Configuration());
            BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(pt, true)));

            for (int i = 0; i < nV; i++) {
                for (int j = 0; j < nF; j++) {
                    br.write(data[i * nF + j] + " ");
                }
                br.write("\n");
            }
            br.close();
        }
        catch(Exception e) {
            System.out.println("File not found");
        }
    }

}
