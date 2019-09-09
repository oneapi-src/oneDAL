/* file: KmeansDenseInitStep1Mapper.java */
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
import com.intel.daal.algorithms.kmeans.init.*;
import com.intel.daal.algorithms.kmeans.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class KmeansDenseInitStep1Mapper extends Mapper<Object, Text, IntWritable, WriteableData> {

    private static final int nBlocks = 4;
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
                    Context context) throws IOException, InterruptedException {

        DaalContext daalContext = new DaalContext();

        /* Read a data set */
        String filePath = "/Hadoop/KmeansDense/data/" + value;
        double[] data = new double[nFeatures * nVectorsInBlock];
        readData(filePath, nFeatures, nVectorsInBlock, data);

        HomogenNumericTable ntData = new HomogenNumericTable(daalContext, data, nFeatures, nVectorsInBlock);

        /* Create an algorithm to initialize the K-Means algorithm on local nodes */
        InitDistributedStep1Local kmeansLocalInit = new InitDistributedStep1Local(daalContext, Double.class, InitMethod.randomDense,
                                                                                  nClusters, nVectorsInBlock * nBlocks, nVectorsInBlock * index);
        kmeansLocalInit.input.set( InitInputId.data, ntData );

        /* Initialize the K-Means algorithm on local nodes */
        InitPartialResult pres = kmeansLocalInit.compute();

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
