/* file: SVD.java */
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
 //     Java sample of computing singular value decomposition (SVD) in the
 //     distributed processing mode
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
import com.intel.daal.services.*;

/* Implement Tool to be able to pass -libjars on start */
public class SVD extends Configured implements Tool {

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new SVD(), args);
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

        Job job = new Job(conf, "SVD Job (step1 and step2)");

        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path("/Hadoop/SVD/step2"));

        job.setMapperClass(SVDStep1Mapper.class);
        job.setReducerClass(SVDStep2Reducer.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(WriteableData.class);

        job.setJarByClass(SVD.class);

        job.waitForCompletion(true);

        Job job1 = new Job(conf, "SVD Job (step3)");

        FileInputFormat.setInputPaths(job1, new Path("/Hadoop/SVD/step2"));
        FileOutputFormat.setOutputPath(job1, new Path(args[1]));

        job1.setMapperClass(SVDStep3Mapper.class);
        job1.setNumReduceTasks(0);

        job1.setInputFormatClass(SequenceFileInputFormat.class);

        job1.setJarByClass(SVD.class);

        return job1.waitForCompletion(true) ? 0 : 1;

    }
}
