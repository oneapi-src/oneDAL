/* file: LossForwardInputId.java */
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
 * @ingroup loss_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSFORWARDINPUTID"></a>
 * \brief Available identifiers of input objects for the forward layer
 */
public final class LossForwardInputId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public LossForwardInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int dataId = 0;
    private static final int weightsId = 1;
    private static final int biasesId = 2;
    private static final int groundTruthId = 4;

    public static final LossForwardInputId data = new LossForwardInputId(dataId); /*!< Input data */
    public static final LossForwardInputId weights = new LossForwardInputId(weightsId); /*!< Weights of the neural network layer */
    public static final LossForwardInputId biases = new LossForwardInputId(biasesId); /*!< Biases of the neural network layer */
    public static final LossForwardInputId groundTruth = new LossForwardInputId(groundTruthId); /*!< Ground truth of the neural network layer */
}
/** @} */
