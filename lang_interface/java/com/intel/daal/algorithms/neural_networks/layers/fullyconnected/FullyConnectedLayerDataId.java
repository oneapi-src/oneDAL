/* file: FullyConnectedLayerDataId.java */
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
 * @ingroup fullyconnected
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.fullyconnected;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__FULLYCONNECTEDLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward fully-connected layer and results for the forward fully-connected layer
 */
public final class FullyConnectedLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public FullyConnectedLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int auxDataId = 0;
    private static final int auxWeightsId = 1;

    public static final FullyConnectedLayerDataId auxData    = new FullyConnectedLayerDataId(auxDataId);     /*!< Data processed at the forward stage of the layer */
    public static final FullyConnectedLayerDataId auxWeights = new FullyConnectedLayerDataId(auxWeightsId);  /*!< Weights of the 2D convolution layer */
}
/** @} */
