/* file: SparkQR.java */
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
 //     Java sample of computing QR decomposition in the distributed processing
 //     mode
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

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.algorithms.qr.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.services.*;

public class SparkQR {
    /* Class containing the algorithm results */
    static class QRResult {
        public HomogenNumericTable R;
        public JavaPairRDD<Integer, HomogenNumericTable> Q;
    }

    static QRResult result = new QRResult();

    static JavaPairRDD<Integer, DataCollection> dataFromStep1ForStep2_RDD;
    static JavaPairRDD<Integer, DataCollection> dataFromStep1ForStep3_RDD;
    static JavaPairRDD<Integer, DataCollection> dataFromStep2ForStep3_RDD;

    public static QRResult runQR(DaalContext context, JavaPairRDD<Integer, HomogenNumericTable> dataRDD, JavaSparkContext sc) {

        computeStep1Local(dataRDD);

        computeStep2Master(context, sc);

        computeStep3Local();

        return result;
    }

    private static void computeStep1Local(JavaPairRDD<Integer, HomogenNumericTable> dataRDD) {
        /* Create an RDD containing partial results for steps 2 and 3 */
        JavaPairRDD<Integer, Tuple2<DataCollection, DataCollection>> dataFromStep1_RDD = dataRDD.mapToPair(
        new PairFunction<Tuple2<Integer, HomogenNumericTable>, Integer, Tuple2<DataCollection, DataCollection>>() {
            public Tuple2<Integer, Tuple2<DataCollection, DataCollection>> call(Tuple2<Integer, HomogenNumericTable> tup) {
                DaalContext context = new DaalContext();

                /* Create an algorithm to compute QR decomposition on local nodes */
                DistributedStep1Local qrStep1Local = new DistributedStep1Local(context, Double.class, Method.defaultDense);
                tup._2.unpack(context);
                qrStep1Local.input.set( InputId.data, tup._2 );

                /* Compute QR decomposition in step 1  */
                DistributedStep1LocalPartialResult pres = qrStep1Local.compute();
                DataCollection dataFromStep1ForStep2 = pres.get( PartialResultId.outputOfStep1ForStep2 );
                dataFromStep1ForStep2.pack();
                DataCollection dataFromStep1ForStep3 = pres.get( PartialResultId.outputOfStep1ForStep3 );
                dataFromStep1ForStep3.pack();

                context.dispose();

                return new Tuple2<Integer, Tuple2<DataCollection, DataCollection>>(
                           tup._1, new Tuple2<DataCollection, DataCollection>(dataFromStep1ForStep2, dataFromStep1ForStep3));
            }
        }).cache();

        /* Extract partial results for step 3 */
        dataFromStep1ForStep3_RDD = dataFromStep1_RDD.mapToPair(
        new PairFunction<Tuple2<Integer, Tuple2<DataCollection, DataCollection>>, Integer, DataCollection>() {
            public Tuple2<Integer, DataCollection> call(Tuple2<Integer, Tuple2<DataCollection, DataCollection>> tup) {
                return new Tuple2<Integer, DataCollection>(tup._1, tup._2._2);
            }
        });

        /* Extract partial results for step 2 */
        dataFromStep1ForStep2_RDD = dataFromStep1_RDD.mapToPair(
        new PairFunction<Tuple2<Integer, Tuple2<DataCollection, DataCollection>>, Integer, DataCollection>() {
            public Tuple2<Integer, DataCollection> call(Tuple2<Integer, Tuple2<DataCollection, DataCollection>> tup) {
                return new Tuple2<Integer, DataCollection>(tup._1, tup._2._1);
            }
        });
    }

    private static void computeStep2Master(DaalContext context, JavaSparkContext sc) {

        int nBlocks = (int)(dataFromStep1ForStep2_RDD.count());

        List<Tuple2<Integer, DataCollection>> dataFromStep1ForStep2_List = dataFromStep1ForStep2_RDD.collect();

        /* Create an algorithm to compute QR decomposition on the master node */
        DistributedStep2Master qrStep2Master = new DistributedStep2Master( context, Double.class, Method.defaultDense );

        for (Tuple2<Integer, DataCollection> value : dataFromStep1ForStep2_List) {
            value._2.unpack(context);
            qrStep2Master.input.add( DistributedStep2MasterInputId.inputOfStep2FromStep1, value._1, value._2 );
        }

        /* Compute QR decomposition in step 2 */
        DistributedStep2MasterPartialResult pres = qrStep2Master.compute();

        KeyValueDataCollection inputForStep3FromStep2 = pres.get( DistributedPartialResultCollectionId.outputOfStep2ForStep3 );

        List<Tuple2<Integer, DataCollection>> list = new ArrayList<Tuple2<Integer, DataCollection>>(nBlocks);
        for (Tuple2<Integer, DataCollection> value : dataFromStep1ForStep2_List) {
            DataCollection dc = (DataCollection) inputForStep3FromStep2.get( value._1 );
            dc.pack();
            list.add(new Tuple2<Integer, DataCollection>(value._1, dc));
        }

        /* Make PairRDD from the list */
        dataFromStep2ForStep3_RDD = JavaPairRDD.fromJavaRDD( sc.parallelize(list, nBlocks) );

        Result res = qrStep2Master.finalizeCompute();

        HomogenNumericTable ntR = (HomogenNumericTable)res.get( ResultId.matrixR );
        result.R = ntR;
    }

    private static void computeStep3Local() {
        /* Group partial results from steps 1 and 2 */
        JavaPairRDD<Integer, Tuple2<Iterable<DataCollection>, Iterable<DataCollection>>> dataForStep3_RDD =
            dataFromStep1ForStep3_RDD.cogroup(dataFromStep2ForStep3_RDD);

        JavaPairRDD<Integer, HomogenNumericTable> ntQ_RDD = dataForStep3_RDD.mapToPair(
        new PairFunction<Tuple2<Integer, Tuple2<Iterable<DataCollection>, Iterable<DataCollection>>>, Integer, HomogenNumericTable>() {

            public Tuple2<Integer, HomogenNumericTable>
            call(Tuple2<Integer, Tuple2<Iterable<DataCollection>, Iterable<DataCollection>>> tup) {
                DaalContext context = new DaalContext();

                DataCollection ntQPi = tup._2._1.iterator().next();
                ntQPi.unpack(context);
                DataCollection ntPi  = tup._2._2.iterator().next();
                ntPi.unpack(context);

                /* Create an algorithm to compute QR decomposition on the master node */
                DistributedStep3Local qrStep3Local = new DistributedStep3Local(context, Double.class, Method.defaultDense);
                qrStep3Local.input.set( DistributedStep3LocalInputId.inputOfStep3FromStep1, ntQPi );
                qrStep3Local.input.set( DistributedStep3LocalInputId.inputOfStep3FromStep2, ntPi );

                /* Compute QR decomposition in step 3 */
                qrStep3Local.compute();
                Result result = qrStep3Local.finalizeCompute();

                HomogenNumericTable Qi = (HomogenNumericTable)result.get( ResultId.matrixQ );
                Qi.pack();

                context.dispose();

                return new Tuple2<Integer, HomogenNumericTable>(tup._1, Qi);
            }
        });
        result.Q = ntQ_RDD;
    }
}
