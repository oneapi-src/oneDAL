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
 * @defgroup kdtree_knn_classification_training Training
 * @brief Contains a class for KD-tree based kNN model-based training
 * @ingroup kdtree_knn_classification
 * @{
 */
package com.intel.daal.algorithms.kdtree_knn_classification.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__TRAININGMETHOD"></a>
 * @brief Available methods for k nearest neighbors model-based training
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

    private static final int defaultDenseValue = 0;

    public static final TrainingMethod defaultDense = new TrainingMethod(defaultDenseValue);   /*!< Default method */
}
/** @} */
