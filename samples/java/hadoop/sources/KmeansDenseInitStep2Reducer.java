/* file: KmeansDenseInitStep2Reducer.java */
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

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.kmeans.*;
import com.intel.daal.algorithms.kmeans.init.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class KmeansDenseInitStep2Reducer extends Reducer<IntWritable, WriteableData, IntWritable, WriteableData> {

    private static final long nClusters = 20;

    @Override
    public void reduce(IntWritable key, Iterable<WriteableData> values, Context context)
    throws IOException, InterruptedException {

        DaalContext daalContext = new DaalContext();

        /* Create an algorithm to initialize the K-Means algorithm on the master node */
        InitDistributedStep2Master kmeansMasterInit = new InitDistributedStep2Master(daalContext, Double.class, InitMethod.randomDense,
                                                                                     nClusters);
        for (WriteableData value : values) {
            InitPartialResult pr = (InitPartialResult)value.getObject(daalContext);
            kmeansMasterInit.input.add( InitDistributedStep2MasterInputId.partialResults, pr );
        }

        /* Initialize the K-Means algorithm on the master node */
        kmeansMasterInit.compute().dispose();

        /* Finalize computations and retrieve the results */
        InitResult initResult = kmeansMasterInit.finalizeCompute();

        /* Write a centroids table for step 3 */
        HomogenNumericTable centroids = (HomogenNumericTable)initResult.get(InitResultId.centroids);

        SequenceFile.Writer writer = SequenceFile.createWriter(
                                         new Configuration(),
                                         SequenceFile.Writer.file(new Path("/Hadoop/KmeansDense/initResults/centroids")),
                                         SequenceFile.Writer.keyClass(IntWritable.class),
                                         SequenceFile.Writer.valueClass(WriteableData.class));
        writer.append(new IntWritable(0), new WriteableData(centroids));
        writer.close();

        daalContext.dispose();
    }
}
