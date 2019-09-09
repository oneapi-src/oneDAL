/* file: SVDStep3Mapper.java */
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
import com.intel.daal.algorithms.svd.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class SVDStep3Mapper extends Mapper<IntWritable, WriteableData, IntWritable, WriteableData> {

    private static Configuration conf;

    @Override
    public void setup(Context context) {
        conf = context.getConfiguration();
    }

    @Override
    public void map(IntWritable step2key, WriteableData step2value, Context context) throws IOException, InterruptedException {

        DaalContext daalContext = new DaalContext();

        SequenceFile.Reader reader =
            new SequenceFile.Reader( new Configuration(), SequenceFile.Reader.file(new Path("/Hadoop/SVD/step1/step1x" + step2value.getId() )) );
        IntWritable   step1key     = new IntWritable();
        WriteableData step1value   = new WriteableData();
        reader.next(step1key, step1value);
        reader.close();

        DataCollection s1 = (DataCollection)step1value.getObject(daalContext);
        DataCollection s2 = (DataCollection)step2value.getObject(daalContext);

        /* Create an algorithm to compute SVD on the master node */
        DistributedStep3Local svdStep3Local = new DistributedStep3Local(daalContext, Double.class, Method.defaultDense);
        svdStep3Local.input.set( DistributedStep3LocalInputId.inputOfStep3FromStep1, s1 );
        svdStep3Local.input.set( DistributedStep3LocalInputId.inputOfStep3FromStep2, s2 );

        /* Compute SVD in step 3 */
        svdStep3Local.compute();
        Result result = svdStep3Local.finalizeCompute();
        HomogenNumericTable Ui = (HomogenNumericTable)result.get( ResultId.leftSingularMatrix );

        SequenceFile.Writer writer = SequenceFile.createWriter(
                                         new Configuration(),
                                         SequenceFile.Writer.file(new Path("/Hadoop/SVD/Output/Ux" + step2value.getId())),
                                         SequenceFile.Writer.keyClass(IntWritable.class),
                                         SequenceFile.Writer.valueClass(WriteableData.class));
        writer.append( new IntWritable( 0 ), new WriteableData( step2value.getId(), Ui ) );
        writer.close();

        daalContext.dispose();
    }
}
