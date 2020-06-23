/* file: SampleALS.scala */
//==============================================================================
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
