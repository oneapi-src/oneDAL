/* file: SparkKmeansCSR.java */
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
//      Java sample of K-Means clustering in the distributed processing mode
////////////////////////////////////////////////////////////////////////////////
*/

package DAAL;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import scala.Tuple2;
import com.intel.daal.algorithms.kmeans.*;
import com.intel.daal.algorithms.kmeans.init.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

public class SparkKmeansCSR {
    /* Class containing the algorithm results */
    static class KmeansResult {
        public HomogenNumericTable centroids;
    }
    private static final long nBlocks        = 4;
    private static final long nClusters      = 20;
    private static final int nIterations     = 5;
    private static final int nVectorsInBlock = 8000;

    public static KmeansResult runKmeans(DaalContext context, JavaPairRDD<Integer, CSRNumericTable> dataRDD) {

        JavaRDD<InitPartialResult> partsRDD = computeInitLocal(context, dataRDD);
        HomogenNumericTable centroids       = computeInitMaster(context, partsRDD);

        PartialResult partRes;
        JavaRDD<PartialResult> partResRDD;
        for(int it = 0; it < nIterations; it++) {
            centroids.pack();
            partResRDD = computeLocal(context, dataRDD, centroids);
            centroids  = computeMaster(context, partResRDD);
        }

        KmeansResult result = new KmeansResult();
        result.centroids = centroids;
        return result;
    }

    private static JavaRDD<InitPartialResult> computeInitLocal(DaalContext context, JavaPairRDD<Integer, CSRNumericTable> dataRDD) {
        return dataRDD.map(new Function<Tuple2<Integer, CSRNumericTable>, InitPartialResult>() {
            public InitPartialResult call(Tuple2<Integer, CSRNumericTable> tup) {

                DaalContext localContext = new DaalContext();

                /* Create an algorithm to initialize the K-Means algorithm on local nodes */
                InitDistributedStep1Local kmeansLocalInit = new InitDistributedStep1Local(localContext, Double.class, InitMethod.randomCSR,
                                                                                          nClusters, nBlocks * nVectorsInBlock,
                                                                                          nVectorsInBlock * tup._1);
                /* Set the input data on local nodes */
                tup._2.unpack(localContext);
                kmeansLocalInit.input.set(InitInputId.data, tup._2);

                /* Initialize the K-Means algorithm on local nodes */
                InitPartialResult pres = kmeansLocalInit.compute();
                pres.pack();
                tup._2.pack();

                localContext.dispose();
                return pres;
            }
        });
    }

    private static HomogenNumericTable computeInitMaster(DaalContext context, JavaRDD<InitPartialResult> partsRDD) {
        /* Create an algorithm to compute k-means on the master node */
        InitDistributedStep2Master kmeansMasterInit = new InitDistributedStep2Master(context, Double.class,
                                                                                     InitMethod.randomCSR, nClusters);
        /* Set the partial results recieved from the local nodes */
        List<InitPartialResult> partResList = partsRDD.collect();
        for (InitPartialResult value : partResList) {
            if(value != null) {
                value.unpack(context);
                kmeansMasterInit.input.add(InitDistributedStep2MasterInputId.partialResults, value);
            }
        }
        /* Compute initial centroids on the master node */
        kmeansMasterInit.compute();

        /* Finalize computations and retrieve the results */
        InitResult initResult = kmeansMasterInit.finalizeCompute();

        return (HomogenNumericTable)initResult.get(InitResultId.centroids);
    }

    private static JavaRDD<PartialResult> computeLocal(DaalContext context,
                                                       JavaPairRDD<Integer, CSRNumericTable> dataRDD,
                                                       final NumericTable centroids) {
        return dataRDD.map(new Function<Tuple2<Integer, CSRNumericTable>, PartialResult>() {
            public PartialResult call(Tuple2<Integer, CSRNumericTable> tup) {
                DaalContext localContext = new DaalContext();

                /* Create an algorithm to compute k-means on local nodes */
                DistributedStep1Local kmeansLocal = new DistributedStep1Local(localContext, Double.class, Method.lloydCSR, nClusters);

                tup._2.unpack(localContext);
                centroids.unpack(localContext);

                /* Set the input data on local nodes */
                kmeansLocal.input.set(InputId.data, tup._2);
                kmeansLocal.input.set(InputId.inputCentroids, centroids);

                /* Compute k-means on local nodes */
                PartialResult pres = kmeansLocal.compute();

                pres.pack();
                tup._2.pack();
                centroids.pack();

                localContext.dispose();

                return pres;
            }
        });
    }

    private static HomogenNumericTable computeMaster(DaalContext context, JavaRDD<PartialResult> partResRDD) {
        /* Create an algorithm to compute k-means on the master node */
        DistributedStep2Master kmeansMaster = new DistributedStep2Master(context, Double.class, Method.lloydCSR, nClusters);

        /* Set the partial result to the master algorithm to compute the final result */
        List<PartialResult> partResList = partResRDD.collect();
        for (PartialResult value : partResList) {
            if(value != null) {
                value.unpack(context);
                kmeansMaster.input.add(DistributedStep2MasterInputId.partialResults, value);
            }
        }

        /* Compute k-means on the master node */
        kmeansMaster.compute();

        /* Finalize computations and retrieve the results */
        Result res = kmeansMaster.finalizeCompute();
        return (HomogenNumericTable)res.get(ResultId.centroids);
    }
}
