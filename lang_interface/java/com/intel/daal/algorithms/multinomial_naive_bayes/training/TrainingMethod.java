/* file: TrainingMethod.java */
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
 * @defgroup multinomial_naive_bayes_training Training
 * @brief Contains classes for training the naive Bayes model
 * @ingroup multinomial_naive_bayes
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for computing the naive Bayes training results
 */
public final class TrainingMethod {

    private int _value;

    /**
     * Constructs the training method object using the provided value
     * @param value     Value corresponding to the training method object
     */
    public TrainingMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training method object
     * @return Value corresponding to the training method object
     */
    public int getValue() {
        return _value;
    }

    private static final int DefaultDense = 0;
    private static final int FastCSR      = 1;

    public static final TrainingMethod defaultDense = new TrainingMethod(DefaultDense); /*!< Default Multinomial Naive Bayes train method */
    public static final TrainingMethod fastCSR      = new TrainingMethod(FastCSR);      /*!< Training method for the multinomial naive Bayes
                                                                                             with sparse data in CSR format */
}
/** @} */
