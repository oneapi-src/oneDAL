/* file: CovarianceCSRStep2Reducer.java */
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

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class CovarianceCSRStep2Reducer extends Reducer<IntWritable, WriteableData, IntWritable, WriteableData> {

    @Override
    public void reduce(IntWritable key, Iterable<WriteableData> values, Context context)
    throws IOException, InterruptedException {

        DaalContext daalContext = new DaalContext();

        /* Create an algorithm to compute a sparse variance-covariance matrix on the master node */
        DistributedStep2Master covarianceSparseMaster = new DistributedStep2Master(daalContext, Double.class, Method.fastCSR);

        for (WriteableData value : values) {
            PartialResult pr = (PartialResult)value.getObject(daalContext);
            covarianceSparseMaster.input.add( DistributedStep2MasterInputId.partialResults, pr );
        }

        /* Compute a sparse variance-covariance matrix on the master node */
        covarianceSparseMaster.compute();

        /* Finalize computations and retrieve the results */
        Result result = covarianceSparseMaster.finalizeCompute();

        HomogenNumericTable covariance = (HomogenNumericTable)result.get(ResultId.covariance);
        HomogenNumericTable mean       = (HomogenNumericTable)result.get(ResultId.mean);

        context.write(new IntWritable(0), new WriteableData( covariance  ) );
        context.write(new IntWritable(1), new WriteableData( mean ) );

        daalContext.dispose();
    }
}
