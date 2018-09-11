/* file: EluLayerDataId.java */
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
 * @ingroup elu_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.elu;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELU__ELULAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward ELU layer and results for the forward ELU layer
 */
public final class EluLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public EluLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    public static final EluLayerDataId auxData =
        new EluLayerDataId(cGetAuxDataId()); /*!< Data processed at the forward stage of the layer */

    public static final EluLayerDataId auxIntermediateValue =
        new EluLayerDataId(cGetAuxIntermediateValueId()); /*!< Data processed at the forward stage of the layer */

    private static native int cGetAuxDataId();
    private static native int cGetAuxIntermediateValueId();
}
/** @} */
