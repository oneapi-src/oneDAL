/* file: SparkKmeansDense.java */
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

public class SparkKmeansDense {
    /* Class containing the algorithm results */
    static class KmeansResult {
        public HomogenNumericTable centroids;
    }
    private static final long nBlocks        = 4;
    private static final long nClusters      = 20;
    private static final int nIterations     = 5;
    private static final int nVectorsInBlock = 2500;

    public static KmeansResult runKmeans(DaalContext context, JavaPairRDD<Integer, HomogenNumericTable> dataRDD) {

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

    private static JavaRDD<InitPartialResult> computeInitLocal(DaalContext context, JavaPairRDD<Integer, HomogenNumericTable> dataRDD) {
        return dataRDD.map(new Function<Tuple2<Integer, HomogenNumericTable>, InitPartialResult>() {
            public InitPartialResult call(Tuple2<Integer, HomogenNumericTable> tup) {

                DaalContext localContext = new DaalContext();

                /* Create an algorithm to initialize the K-Means algorithm on local nodes */
                InitDistributedStep1Local kmeansLocalInit = new InitDistributedStep1Local(localContext, Double.class, InitMethod.randomDense,
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
                                                                                     InitMethod.randomDense, nClusters);
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
                                                       JavaPairRDD<Integer, HomogenNumericTable> dataRDD,
                                                       final NumericTable centroids) {
        return dataRDD.map(new Function<Tuple2<Integer, HomogenNumericTable>, PartialResult>() {
            public PartialResult call(Tuple2<Integer, HomogenNumericTable> tup) {
                DaalContext localContext = new DaalContext();

                /* Create an algorithm to compute k-means on local nodes */
                DistributedStep1Local kmeansLocal = new DistributedStep1Local(localContext, Double.class, Method.defaultDense, nClusters);

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
        DistributedStep2Master kmeansMaster = new DistributedStep2Master(context, Double.class, Method.defaultDense, nClusters);

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
