/* file: PredictionMethod.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
*******************************************************************************/

/**
 * @defgroup multinomial_naive_bayes_prediction Prediction
 * @brief Contains classes for multinomial naive Bayes model based prediction
 * @ingroup multinomial_naive_bayes
 * @{
 */
/**
 * @brief Contains classes for multinomial naive Bayes prediction method
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PREDICTION__PREDICTIONMETHOD"></a>
 * @brief Available methods for computing the results of the naive Bayes model based prediction
 */
public final class PredictionMethod {

    private int _value;

    /**
     * Constructs the prediction method object using the provided value
     * @param value     Value corresponding to the prediction method object
     */
    public PredictionMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the prediction method object
     * @return Value corresponding to the prediction method object
     */
    public int getValue() {
        return _value;
    }

    private static final int DefaultDense = 0;
    private static final int FastCSR      = 1;

    public static final PredictionMethod defaultDense = new PredictionMethod(DefaultDense); /*!< Default Multinomial Naive Bayes prediction method */
    public static final PredictionMethod fastCSR      = new PredictionMethod(FastCSR);      /*!< Multinomial naive Bayes model based prediction for
                                                                                                 sparse data in CSR format */
}
/** @} */
