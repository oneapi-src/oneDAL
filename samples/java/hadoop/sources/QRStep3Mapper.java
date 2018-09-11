/* file: QRStep3Mapper.java */
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
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.util.StringTokenizer;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.conf.Configuration;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.qr.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class QRStep3Mapper extends Mapper<IntWritable, WriteableData, IntWritable, WriteableData> {

    private static Configuration conf;

    @Override
    public void setup(Context context) {
        conf = context.getConfiguration();
    }

    @Override
    public void map(IntWritable step2key, WriteableData step2value, Context context) throws IOException, InterruptedException {

        DaalContext daalContext = new DaalContext();

        SequenceFile.Reader reader =
            new SequenceFile.Reader( new Configuration(), SequenceFile.Reader.file(new Path("/Hadoop/QR/step1/step1x" + step2value.getId() )) );
        IntWritable   step1key     = new IntWritable();
        WriteableData step1value   = new WriteableData();
        reader.next(step1key, step1value);
        reader.close();

        DataCollection s1 = (DataCollection)step1value.getObject(daalContext);
        DataCollection s2 = (DataCollection)step2value.getObject(daalContext);

        /* Create an algorithm to compute QR decomposition on the master node */
        DistributedStep3Local qrStep3Local = new DistributedStep3Local(daalContext, Double.class, Method.defaultDense);
        qrStep3Local.input.set( DistributedStep3LocalInputId.inputOfStep3FromStep1, s1 );
        qrStep3Local.input.set( DistributedStep3LocalInputId.inputOfStep3FromStep2, s2 );

        /* Compute QR decomposition in step 3 */
        qrStep3Local.compute();
        Result result = qrStep3Local.finalizeCompute();
        HomogenNumericTable Qi = (HomogenNumericTable)result.get( ResultId.matrixQ );

        SequenceFile.Writer writer = SequenceFile.createWriter(
                                         new Configuration(),
                                         SequenceFile.Writer.file(new Path("/Hadoop/QR/Output/Qx" + step2value.getId())),
                                         SequenceFile.Writer.keyClass(IntWritable.class),
                                         SequenceFile.Writer.valueClass(WriteableData.class));
        writer.append( new IntWritable( 0 ), new WriteableData( step2value.getId(), Qi ) );
        writer.close();

        daalContext.dispose();
    }
}
