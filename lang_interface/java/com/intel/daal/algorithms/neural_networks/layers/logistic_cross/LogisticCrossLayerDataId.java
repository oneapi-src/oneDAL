/* file: LogisticCrossLayerDataId.java */
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
 * @ingroup logistic_cross
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC_CROSS__LOGISTICCROSSLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward logistic cross-entropy layer and results for the forward logistic cross-entropy layer
 */
public final class LogisticCrossLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public LogisticCrossLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int auxDataId = 2;
    private static final int auxGroundTruthId = 3;

    public static final LogisticCrossLayerDataId auxData = new LogisticCrossLayerDataId(auxDataId); /*!< Tensor that stores probabilities for the forward logistic cross-entropy layer */
    public static final LogisticCrossLayerDataId auxGroundTruth = new LogisticCrossLayerDataId(auxGroundTruthId); /*!< Tensor that stores ground truth data for the forward logistic cross-entropy layer */
}
/** @} */
