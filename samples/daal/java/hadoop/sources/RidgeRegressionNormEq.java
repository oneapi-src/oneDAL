/* file: RidgeRegressionNormEq.java */
/*******************************************************************************
* Copyright 2017 Intel Corporation
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

/*
 //  Content:
 //     Java sample of ridge regression in the distributed processing mode.
 //
 //     The program trains the ridge regression model on a training
 //     data set with the normal equations method and computes regression for
 //     the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

package DAAL;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import java.net.URI;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

/* Implement Tool to be able to pass -libjars on start */
public class RidgeRegressionNormEq extends Configured implements Tool {

    public static void main(String[] args) {
        int res = -1;
        try {
            res = ToolRunner.run(new Configuration(), new RidgeRegressionNormEq(), args);
        } catch (Exception e) {
            ErrorHandling.printThrowable(e);
        }
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = this.getConf();

        /* Put shared libraries into the distributed cache */
        DistributedCache.createSymlink(conf);
        DistributedCache.addCacheFile(new URI("/Hadoop/Libraries/" + System.getenv("LIBJAVAAPI")), conf);
        DistributedCache.addCacheFile(new URI("/Hadoop/Libraries/" + System.getenv("LIBTBB")), conf);
        DistributedCache.addCacheFile(new URI("/Hadoop/Libraries/" + System.getenv("LIBTBBMALLOC")), conf);

        Job job = new Job(conf, "Ridge regression with normal equations method (normEq) Job");

        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setMapperClass(RidgeRegressionNormEqStep1TrainingMapper.class);
        job.setReducerClass(RidgeRegressionNormEqStep2TrainingReducerAndPrediction.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(WriteableData.class);
        job.setJarByClass(RidgeRegressionNormEq.class);

        return job.waitForCompletion(true) ? 0 : 1;
    }
}
