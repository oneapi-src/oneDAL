/* file: LcnLayerDataId.java */
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
 * @ingroup lcn_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward local contrast normalization layer and results for the forward local contrast normalization layer
 */
public final class LcnLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public LcnLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int auxCenteredDataId = 0;
    private static final int auxSigmaId        = 1;
    private static final int auxCId            = 2;
    private static final int auxInvMaxId       = 3;

    public static final LcnLayerDataId auxCenteredData = new LcnLayerDataId(auxCenteredDataId);
    public static final LcnLayerDataId auxSigma        = new LcnLayerDataId(auxSigmaId);
    public static final LcnLayerDataId auxC            = new LcnLayerDataId(auxCId);
    public static final LcnLayerDataId auxInvMax       = new LcnLayerDataId(auxInvMaxId);
}
/** @} */
