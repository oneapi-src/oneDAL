/* file: SpatialStochasticPooling2dBackwardInput.java */
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
 * @defgroup spatial_stochastic_pooling2d_backward Backward Two-dimensional Spatial pyramid stochastic Pooling Layer
 * @brief Contains classes for backward spatial pyramid stochastic 2D pooling layer
 * @ingroup spatial_stochastic_pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_stochastic_pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_STOCHASTIC_POOLING2D__SPATIALSTOCHASTICPOOLING2DBACKWARDINPUT"></a>
 * @brief Input object for the backward two-dimensional spatial stochastic pooling layer
 */
public final class SpatialStochasticPooling2dBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dBackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SpatialStochasticPooling2dBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward two-dimensional spatial stochastic pooling layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(SpatialStochasticPooling2dLayerDataId id, Tensor val) {
        if (id == SpatialStochasticPooling2dLayerDataId.auxSelectedIndices) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect SpatialStochasticPooling2dBackwardInputId");
        }
    }

    /**
     * Sets the input object of the backward two-dimensional spatial stochastic pooling layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(SpatialStochasticPooling2dLayerDataNumericTableId id, NumericTable val) {
        if (id == SpatialStochasticPooling2dLayerDataNumericTableId.auxInputDimensions) {
            cSetInputNumericTable(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect SpatialStochasticPooling2dBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward two-dimensional spatial stochastic pooling layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(SpatialStochasticPooling2dLayerDataId id) {
        if (id == SpatialStochasticPooling2dLayerDataId.auxSelectedIndices) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object of the backward two-dimensional spatial stochastic pooling layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(SpatialStochasticPooling2dLayerDataNumericTableId id) {
        if (id == SpatialStochasticPooling2dLayerDataNumericTableId.auxInputDimensions) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputNumericTable(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
    private native void cSetInputNumericTable(long cObject, int id, long ntAddr);
    private native long cGetInputNumericTable(long cObject, int id);
}
/** @} */
