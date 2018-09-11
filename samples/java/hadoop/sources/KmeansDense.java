/* file: KmeansDense.java */
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

/*
 //  Content:
 //     Java sample of dense K-Means clustering in the distributed processing mode
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
public class KmeansDense extends Configured implements Tool {

    private static final int nIterations = 5;

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new KmeansDense(), args);
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

        Job initJob = new Job(conf, "K-means Init Job");

        FileInputFormat.setInputPaths(initJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(initJob, new Path("/Hadoop/KmeansDense/initResults"));

        initJob.setMapperClass(KmeansDenseInitStep1Mapper.class);
        initJob.setReducerClass(KmeansDenseInitStep2Reducer.class);
        initJob.setInputFormatClass(TextInputFormat.class);
        initJob.setOutputFormatClass(SequenceFileOutputFormat.class);
        initJob.setOutputKeyClass(IntWritable.class);
        initJob.setOutputValueClass(WriteableData.class);
        initJob.setJarByClass(KmeansDense.class);
        initJob.waitForCompletion(true);

        int a = 1;

        for(int i = 0; i < nIterations; i++) {
            Job job = new Job(conf, "K-means compute Job");
            FileInputFormat.setInputPaths(job, new Path(args[0]));

            if( i == nIterations - 1) {
                FileOutputFormat.setOutputPath(job, new Path(args[1]));
            }
            else {
                FileOutputFormat.setOutputPath(job, new Path("/Hadoop/KmeansDense/ResultsIter" + i));
            }
            job.setMapperClass(KmeansDenseStep1Mapper.class);
            job.setReducerClass(KmeansDenseStep2Reducer.class);
            job.setInputFormatClass(TextInputFormat.class);
            job.setOutputFormatClass(SequenceFileOutputFormat.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(WriteableData.class);
            job.setJarByClass(KmeansDense.class);

            if (job.waitForCompletion(true)) { a = 0; }
            else { a = 1; };
        }

        return a;
    }
}
