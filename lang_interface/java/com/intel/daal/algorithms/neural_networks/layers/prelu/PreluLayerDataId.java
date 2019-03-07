/* file: PreluLayerDataId.java */
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
 * @ingroup prelu
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.prelu;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__PRELULAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward prelu layer and results for the forward prelu layer
 */
public final class PreluLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public PreluLayerDataId(int value) {
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

    public static final PreluLayerDataId auxData    = new PreluLayerDataId(auxDataId);    /*!< Data processed at the forward stage of the layer */
    public static final PreluLayerDataId auxWeights = new PreluLayerDataId(auxWeightsId); /*!< Weights of the prelu layer */
}
/** @} */
