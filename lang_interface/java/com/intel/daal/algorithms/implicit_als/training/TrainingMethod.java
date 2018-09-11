/* file: TrainingMethod.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup implicit_als_training
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for training the implicit ALS model
 */
public final class TrainingMethod {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

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

    private static final int defaultDenseId = 0;
    private static final int fastCSRId      = 1;

    /**
    * Method for training the implicit ALS model
    */
    public static final TrainingMethod defaultDense = new TrainingMethod(
            defaultDenseId);    /*!< Default: method proposed by Hu, Koren,
                                    Volinsky for input data stored in the dense format */
    /**
    * Method for training the implicit ALS model
    */
    public static final TrainingMethod fastCSR      = new TrainingMethod(
            fastCSRId);         /*!< Method proposed by Hu, Koren,
                                    Volinsky for input data stored in the compressed sparse row (CSR) format */
}
/** @} */
