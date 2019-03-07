/* file: MaximumPooling2dLayerDataId.java */
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
 * @ingroup maximum_pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling2d;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING2D__MAXIMUMPOOLING2DLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward two-dimensional maximum pooling layer and
 *        results for the forward two-dimensional maximum pooling layer
 */
public final class MaximumPooling2dLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public MaximumPooling2dLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int auxSelectedIndicesId = 0;

    public static final MaximumPooling2dLayerDataId auxSelectedIndices = new MaximumPooling2dLayerDataId(
        auxSelectedIndicesId);    /*!< p-dimensional tensor that stores the positions of maximum elements */
}
/** @} */
