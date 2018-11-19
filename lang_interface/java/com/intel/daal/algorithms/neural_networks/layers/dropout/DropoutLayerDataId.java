/* file: DropoutLayerDataId.java */
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
 * @ingroup dropout
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.dropout;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__DROPOUT__DROPOUTLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward dropout layer and results for the forward dropout layer
 */
public final class DropoutLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public DropoutLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int auxRetainMaskId = 2;

    public static final DropoutLayerDataId auxRetainMask = new DropoutLayerDataId(auxRetainMaskId);
            /*!< Tensor filled with Bernoulli random variates  (0 in positions that are dropped,
                 1 - in the others) divided by probability that any particular element is retained. */
}
/** @} */
