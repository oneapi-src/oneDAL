/* file: KmeansCSRStep2Reducer.java */
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
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class KmeansCSRStep2Reducer extends Reducer<IntWritable, WriteableData, IntWritable, WriteableData> {

    private static final long nClusters = 20;

    @Override
    public void reduce(IntWritable key, Iterable<WriteableData> values, Context context)
    throws IOException, InterruptedException {

        /* Create an algorithm to compute k-means on the master node */
        DaalContext daalContext = new DaalContext();

        DistributedStep2Master kmeansMaster = new DistributedStep2Master(daalContext, Double.class, Method.lloydCSR,
                                                                         nClusters);

        for (WriteableData value : values) {
            PartialResult pr = (PartialResult)value.getObject(daalContext);
            kmeansMaster.input.add(DistributedStep2MasterInputId.partialResults, pr);
            pr.dispose();
        }

        /* Compute k-means on the master node */
        kmeansMaster.compute().dispose();

        /* Finalize computations and retrieve the results */
        Result result = kmeansMaster.finalizeCompute();

        HomogenNumericTable centroids = (HomogenNumericTable)result.get(ResultId.centroids);

        context.write(new IntWritable(0), new WriteableData(centroids));

        /* Write centroids for the next iteration */
        SequenceFile.Writer writer = SequenceFile.createWriter(
                                         new Configuration(),
                                         SequenceFile.Writer.file(new Path("/Hadoop/KmeansCSR/initResults/centroids")),
                                         SequenceFile.Writer.keyClass(IntWritable.class),
                                         SequenceFile.Writer.valueClass(WriteableData.class));
        writer.append(new IntWritable(0), new WriteableData(centroids));
        writer.close();

        daalContext.dispose();
    }
}
