/* file: LrnLayerDataId.java */
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
 * @ingroup lrn
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lrn;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__LRNLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward local response normalization layer and results for the forward
  *       local response normalization layer
 */
public final class LrnLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public LrnLayerDataId(int value) {
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
    private static final int auxSmBetaId = 1;

    public static final LrnLayerDataId auxData   = new LrnLayerDataId(auxDataId);    /*!< Data processed at the forward stage of the layer */
    public static final LrnLayerDataId auxSmBeta = new LrnLayerDataId(auxSmBetaId);  /*!< Pointer to the tensor of size n1 x n2 x ... x np, that stores
                                                                                    value of (kappa + alpha * sum((x_i)^2))^(-beta) */
}
/** @} */
