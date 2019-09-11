/* file: PredictionInput.java */
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
 * @defgroup neural_networks_prediction Prediction
 * @brief Contains classes for making prediction based on the trained model
 * @ingroup neural_networks
 * @{
 */
package com.intel.daal.algorithms.neural_networks.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__PREDICTIONINPUT"></a>
 * @brief Input objects of the neural networks prediction algorithm
 */
public class PredictionInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PredictionInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object for the neural networks prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(PredictionTensorInputId id, Tensor val) {
        if (id == PredictionTensorInputId.data) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect PredictionTensorInputId");
        }
    }

    /**
     * Sets the input object for the neural networks prediction algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(PredictionModelInputId id, PredictionModel val) {
        if (id == PredictionModelInputId.model) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect PredictionModelInputId");
        }
    }

    /**
     * Returns input object for the neural networks prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(PredictionTensorInputId id) {
        if (id == PredictionTensorInputId.data) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns input object for the neural networks prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public PredictionModel get(PredictionModelInputId id) {
        if (id == PredictionModelInputId.model) {
            return new PredictionModel(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
