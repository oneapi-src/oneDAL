/* file: SparkLowOrderMomentsCSR.java */
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
//      Java sample of computing low order moments in the distributed
//      processing mode.
//
//      Input matrix is stored in the compressed sparse row (CSR) format.
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

import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class SparkLowOrderMomentsCSR {
    /* Class containing the algorithm results */
    static class MomentsResult {
        public HomogenNumericTable minimum;
        public HomogenNumericTable maximum;
        public HomogenNumericTable sum;
        public HomogenNumericTable sumSquares;
        public HomogenNumericTable sumSquaresCentered;
        public HomogenNumericTable mean;
        public HomogenNumericTable secondOrderRawMoment;
        public HomogenNumericTable variance;
        public HomogenNumericTable standardDeviation;
        public HomogenNumericTable variation;
    }

    static JavaPairRDD<Integer, PartialResult> partsRDD;

    public static MomentsResult runMoments(DaalContext context, JavaRDD<CSRNumericTable> dataRDD) {
        JavaRDD<PartialResult> partsRDD = computestep1Local(dataRDD);

        PartialResult finalPartRes = reducePartialResults(partsRDD);

        MomentsResult result = finalizeMergeOnMasterNode(context, finalPartRes);

        return result;
    }

    private static JavaRDD<PartialResult> computestep1Local(JavaRDD<CSRNumericTable> dataRDD) {
        return dataRDD.map(new Function<CSRNumericTable, PartialResult>() {
            public PartialResult call(CSRNumericTable table) {
                DaalContext context = new DaalContext();

                /* Create an algorithm to compute low order moments on local nodes */
                DistributedStep1Local momentsLocal = new DistributedStep1Local(context, Double.class, Method.fastCSR);

                /* Set the input data on local nodes */
                table.unpack(context);
                momentsLocal.input.set( InputId.data, table );

                /* Compute low order moments on local nodes */
                PartialResult pres = momentsLocal.compute();
                pres.pack();

                context.dispose();
                return pres;
            }
        });
    }

    private static PartialResult reducePartialResults(JavaRDD<PartialResult> partsRDD) {
        return partsRDD.reduce(new Function2<PartialResult, PartialResult, PartialResult>() {
            public PartialResult call(PartialResult p1, PartialResult p2) {
                DaalContext context = new DaalContext();

                /* Create an algorithm to compute new partial result from two partial results */
                DistributedStep2Master momentsMaster = new DistributedStep2Master(context, Double.class, Method.fastCSR);

                /* Set the partial results recieved from the local nodes */
                p1.unpack(context);
                p2.unpack(context);
                momentsMaster.input.add(DistributedStep2MasterInputId.partialResults, p1);
                momentsMaster.input.add(DistributedStep2MasterInputId.partialResults, p2);

                /* Compute a new partial result from two partial results */
                PartialResult pres = momentsMaster.compute();
                pres.pack();

                context.dispose();
                return pres;
            }
        });
    }

    private static MomentsResult finalizeMergeOnMasterNode(DaalContext context, PartialResult partRes) {

        /* Create an algorithm to compute low order moments on the master node */
        DistributedStep2Master momentsMaster = new DistributedStep2Master(context, Double.class, Method.fastCSR);

        /* Add partial results computed on local nodes to the algorithm on the master node */
        partRes.unpack(context);
        momentsMaster.input.add(DistributedStep2MasterInputId.partialResults, partRes);

        /* Compute low order moments on the master node */
        momentsMaster.compute();

        /* Finalize computations and retrieve the results */
        Result res = momentsMaster.finalizeCompute();

        MomentsResult result = new MomentsResult();
        result.minimum              = (HomogenNumericTable)res.get(ResultId.minimum);
        result.maximum              = (HomogenNumericTable)res.get(ResultId.maximum);
        result.sum                  = (HomogenNumericTable)res.get(ResultId.sum);
        result.sumSquares           = (HomogenNumericTable)res.get(ResultId.sumSquares);
        result.sumSquaresCentered   = (HomogenNumericTable)res.get(ResultId.sumSquaresCentered);
        result.mean                 = (HomogenNumericTable)res.get(ResultId.mean);
        result.secondOrderRawMoment = (HomogenNumericTable)res.get(ResultId.secondOrderRawMoment);
        result.variance             = (HomogenNumericTable)res.get(ResultId.variance);
        result.standardDeviation    = (HomogenNumericTable)res.get(ResultId.standardDeviation);
        result.variation            = (HomogenNumericTable)res.get(ResultId.variation);
        return result;
    }
}
