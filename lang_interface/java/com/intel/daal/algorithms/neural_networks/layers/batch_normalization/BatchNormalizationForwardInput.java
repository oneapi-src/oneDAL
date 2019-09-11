/* file: BatchNormalizationForwardInput.java */
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
 * @defgroup batch_normalization_forward Forward Batch Normalization Layer
 * @brief Contains classes for forward batch normalization layer
 * @ingroup batch_normalization
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONFORWARDINPUT"></a>
 * @brief %Input object for the forward batch normalization layer
 */
public class BatchNormalizationForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public BatchNormalizationForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the forward batch normalization layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(BatchNormalizationForwardInputLayerDataId id, Tensor val) {
        if (id == BatchNormalizationForwardInputLayerDataId.populationMean || id == BatchNormalizationForwardInputLayerDataId.populationVariance) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect BatchNormalizationForwardInputLayerDataId");
        }
    }

    /**
     * Returns the input object of the forward batch normalization layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(BatchNormalizationForwardInputLayerDataId id) {
        if (id == BatchNormalizationForwardInputLayerDataId.populationMean || id == BatchNormalizationForwardInputLayerDataId.populationVariance) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns dimensions of weights tensor
     * @param parameter  Layer parameter
     * @return Dimensions of weights tensor
     */
    public long[] getWeightsSizes(BatchNormalizationParameter parameter)
    {
        return cGetWeightsSizes(cObject, parameter.getCObject());
    }

    /**
     * Returns dimensions of biases tensor
     * @param parameter  Layer parameter
     * @return Dimensions of biases tensor
     */
    public long[] getBiasesSizes(BatchNormalizationParameter parameter)
    {
        return cGetBiasesSizes(cObject, parameter.getCObject());
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
    private native long[] cGetWeightsSizes(long cObject, long cParameter);
    private native long[] cGetBiasesSizes(long cObject, long cParameter);
}
/** @} */
