/* file: SparkPcaSvd.java */
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
//      Java sample of principal component analysis (PCA) using the singular
//      value decomposition (SVD) method in the distributed processing mode
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
import com.intel.daal.algorithms.pca.*;
import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class SparkPcaSvd {
    /* Class containing the algorithm results */
    static class PCAResult {
        public HomogenNumericTable eigenVectors;
        public HomogenNumericTable eigenValues;
    }

    public static PCAResult runPCA(DaalContext context, JavaRDD<HomogenNumericTable> dataRDD) {
        JavaRDD<PartialResult> partsRDD = computestep1Local(dataRDD);

        PCAResult result = finalizeMergeOnMasterNode(context, partsRDD);

        return result;
    }

    private static JavaRDD<PartialResult> computestep1Local(JavaRDD<HomogenNumericTable> dataRDD) {
        return dataRDD.map(new Function<HomogenNumericTable, PartialResult>() {
            public PartialResult call(HomogenNumericTable table) {
                DaalContext context = new DaalContext();

                /* Create an algorithm to compute PCA decomposition using the correlation method on local nodes */
                DistributedStep1Local pcaLocal = new DistributedStep1Local(context, Double.class, Method.svdDense);

                /* Set the input data on local nodes */
                table.unpack(context);
                pcaLocal.input.set( InputId.data, table );

                /* Compute PCA decomposition on local nodes */
                PartialResult pres = pcaLocal.compute();
                pres.pack();

                context.dispose();
                return pres;
            }
        });
    }

    private static PCAResult finalizeMergeOnMasterNode(DaalContext context, JavaRDD<PartialResult> partsRDD) {

        /* Create an algorithm to compute PCA decomposition using the correlation method on the master node */
        DistributedStep2Master pcaMaster = new DistributedStep2Master(context, Double.class, Method.svdDense);

        /* Add partial results computed on local nodes to the algorithm on the master node */
        List<PartialResult> collectedPres = partsRDD.collect();
        for (PartialResult value : collectedPres) {
            value.unpack(context);
            pcaMaster.input.add(MasterInputId.partialResults, value);
        }

        /* Compute PCA decomposition on the master node */
        pcaMaster.compute();

        /* Finalize computations and retrieve the results */
        Result res = pcaMaster.finalizeCompute();

        PCAResult result = new PCAResult();
        result.eigenVectors = (HomogenNumericTable)res.get(ResultId.eigenVectors);
        result.eigenValues  = (HomogenNumericTable)res.get(ResultId.eigenValues);
        return result;
    }
}
