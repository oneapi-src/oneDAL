/* file: SpatialAveragePooling2dForwardResult.java */
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
 * @ingroup spatial_average_pooling2d_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_average_pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__SPATIALAVERAGEPOOLING2DFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method
 *        of the forward two-dimensional spatial average pooling layer
 */
public final class SpatialAveragePooling2dForwardResult extends com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dForwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SpatialAveragePooling2dForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    * Constructs the forward two-dimensional spatial average pooling layer
    * @param context   Context to manage the forward two-dimensional spatial average pooling layer
    */
    public SpatialAveragePooling2dForwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
     * Returns the result of the forward two-dimensional spatial average pooling layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public NumericTable get(SpatialAveragePooling2dLayerDataId id) {
        if (id == SpatialAveragePooling2dLayerDataId.auxInputDimensions) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the forward two-dimensional spatial average pooling layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(SpatialAveragePooling2dLayerDataId id, NumericTable val) {
        if (id == SpatialAveragePooling2dLayerDataId.auxInputDimensions) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();
    private native long cGetValue(long cObject, int id);
    private native void cSetValue(long cObject, int id, long ntAddr);
}
/** @} */
