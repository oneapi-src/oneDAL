/* file: SampleALS.scala */
//==============================================================================
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,  and
// your use of  them is governed  by the express  license under which  they were
// provided to you (License).  Unless  the License  provides otherwise,  you may
// not use,  modify,  copy,  publish,  distribute,   disclose  or transmit  this
// software or the related documents without Intel's prior written permission.
//
// This software and the related documents are provided  as is,  with no express
// or implied  warranties,  other than  those that are  expressly stated  in the
// License.
//
// License:
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-ag
// reement/
//==============================================================================

package DAAL

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

//import org.apache.spark.mllib.recommendation.ALS
import daal_for_mllib.ALS

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel

import java.io._

object SampleALS extends App {
    val conf = new SparkConf().setAppName("Spark ALS")
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Spark/ALS/data/ALS.txt")
    val ratings = data.map(_.split(' ') match { case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, rate.toDouble)
    })

    // Build the recommendation model using ALS
    val users = 1000
    val products = 1000
    val rank = 10
    val numIterations = 10
    val model = ALS.trainImplicit(users, products, ratings, rank, numIterations)

    // Evaluate the model on rating data
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }

    val predictions = model.predict(usersProducts).map { case Rating(user, product, rate) => ((user, product), rate) }
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) => ((user, product), rate)}.join(predictions)
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()

    println("Mean Squared Error = " + MSE)

    sc.stop()
}
