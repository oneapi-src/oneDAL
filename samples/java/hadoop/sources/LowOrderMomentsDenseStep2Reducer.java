/* file: LowOrderMomentsDenseStep2Reducer.java */
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
import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class LowOrderMomentsDenseStep2Reducer extends Reducer<IntWritable, WriteableData, IntWritable, WriteableData> {

    @Override
    public void reduce(IntWritable key, Iterable<WriteableData> values, Context context)
    throws IOException, InterruptedException {

        DaalContext daalContext = new DaalContext();

        /* Create an algorithm to compute low order moments on the master node */
        DistributedStep2Master momentsDenseMaster = new DistributedStep2Master(daalContext, Double.class, Method.defaultDense);

        for (WriteableData value : values) {
            PartialResult pr = (PartialResult)value.getObject(daalContext);
            momentsDenseMaster.input.add( DistributedStep2MasterInputId.partialResults, pr );
        }

        /* Compute low order moments on the master node */
        momentsDenseMaster.compute();

        /* Finalize computations and retrieve the results */
        Result result = momentsDenseMaster.finalizeCompute();

        HomogenNumericTable minimum              = (HomogenNumericTable)result.get(ResultId.minimum);
        HomogenNumericTable maximum              = (HomogenNumericTable)result.get(ResultId.maximum);
        HomogenNumericTable sum                  = (HomogenNumericTable)result.get(ResultId.sum);
        HomogenNumericTable sumSquares           = (HomogenNumericTable)result.get(ResultId.sumSquares);
        HomogenNumericTable sumSquaresCentered   = (HomogenNumericTable)result.get(ResultId.sumSquaresCentered);
        HomogenNumericTable mean                 = (HomogenNumericTable)result.get(ResultId.mean);
        HomogenNumericTable secondOrderRawMoment = (HomogenNumericTable)result.get(ResultId.secondOrderRawMoment);
        HomogenNumericTable variance             = (HomogenNumericTable)result.get(ResultId.variance);
        HomogenNumericTable standardDeviation    = (HomogenNumericTable)result.get(ResultId.standardDeviation);
        HomogenNumericTable variation            = (HomogenNumericTable)result.get(ResultId.variation);

        context.write(new IntWritable(0), new WriteableData( minimum  ) );
        context.write(new IntWritable(1), new WriteableData( maximum ) );
        context.write(new IntWritable(2), new WriteableData( sum ) );
        context.write(new IntWritable(3), new WriteableData( sumSquares ) );
        context.write(new IntWritable(4), new WriteableData( sumSquaresCentered ) );
        context.write(new IntWritable(5), new WriteableData( mean ) );
        context.write(new IntWritable(6), new WriteableData( secondOrderRawMoment ) );
        context.write(new IntWritable(7), new WriteableData( variance ) );
        context.write(new IntWritable(8), new WriteableData( standardDeviation ) );
        context.write(new IntWritable(9), new WriteableData( variation ) );

        daalContext.dispose();
    }
}
