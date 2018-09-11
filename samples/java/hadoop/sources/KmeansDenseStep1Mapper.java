/* file: KmeansDenseStep1Mapper.java */
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
import java.util.Arrays;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.conf.Configuration;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.kmeans.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class KmeansDenseStep1Mapper extends Mapper<Object, Text, IntWritable, WriteableData> {

    private static final int nFeatures = 20;
    private static final int nVectorsInBlock = 10000;
    private static final long nClusters = 20;

    /* Index is supposed to be a sequence number for the split */
    private int index = 0;
    private int i = 0;
    private int totalTasks = 0;

    @Override
    public void setup(Context context) {
        index = context.getTaskAttemptID().getTaskID().getId();
        Configuration conf = context.getConfiguration();
        totalTasks = conf.getInt("mapred.map.tasks", 0);
    }

    @Override
    public void map(Object key, Text value,
                    Context context) throws IOException,
        InterruptedException {

        /* Read a data set */
        String filePath = "/Hadoop/KmeansDense/data/" + value;
        double[] data = new double[nFeatures * nVectorsInBlock];
        readData(filePath, nFeatures, nVectorsInBlock, data);

        DaalContext daalContext = new DaalContext();

        HomogenNumericTable ntData = new HomogenNumericTable(daalContext, data, nFeatures, nVectorsInBlock);

        /* Create an algorithm to compute k-means on local nodes */
        DistributedStep1Local kmeansLocal = new DistributedStep1Local(daalContext, Double.class, Method.defaultDense, nClusters);

        /* Get the centroids table computed in step 2 */
        SequenceFile.Reader reader =
            new SequenceFile.Reader( new Configuration(),
                                     SequenceFile.Reader.file(new Path("/Hadoop/KmeansDense/initResults/centroids")) );
        IntWritable   step1key     = new IntWritable();
        WriteableData step1value   = new WriteableData();
        reader.next(step1key, step1value);
        reader.close();

        HomogenNumericTable centroids = (HomogenNumericTable)step1value.getObject(daalContext);

        /* Set the algorithm parameters */
        kmeansLocal.input.set(InputId.data, ntData);
        kmeansLocal.input.set(InputId.inputCentroids, centroids);

        /* Compute k-means on local nodes */
        PartialResult pres = kmeansLocal.compute();

        /* Write the data prepended with a data set sequence number. Needed to know the position of the data set in the input data */
        context.write(new IntWritable(0), new WriteableData(index, pres));

        daalContext.dispose();
        index += totalTasks;
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
