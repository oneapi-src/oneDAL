/* file: AveragePooling3dBackwardInput.java */
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
 * @defgroup average_pooling3d_backward Backward Three-dimensional Average Pooling Layer
 * @brief Contains classes for backward average 3D pooling layer
 * @ingroup average_pooling3d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.average_pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING3D__AVERAGEPOOLING3DBACKWARDINPUT"></a>
 * @brief Input object for the backward three-dimensional average pooling layer
 */
public final class AveragePooling3dBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.pooling3d.Pooling3dBackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public AveragePooling3dBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward three-dimensional average pooling layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(AveragePooling3dLayerDataId id, NumericTable val) {
        if (id == AveragePooling3dLayerDataId.auxInputDimensions) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect AveragePooling3dBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward three-dimensional average pooling layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(AveragePooling3dLayerDataId id) {
        if (id == AveragePooling3dLayerDataId.auxInputDimensions) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
