/* file: BackwardLayer.java */
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
 * @defgroup layers_backward Backward Base Layer
 * @brief Contains classes for the backward stage of the neural network layer
 * @ingroup layers
 * @{
 */
/**
 * @defgroup layers_backward_batch Batch
 * @ingroup layers_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARDLAYER"></a>
 * \brief Class representing a backward layer of neural network
 */
public class BackwardLayer extends com.intel.daal.algorithms.AnalysisBatch {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward layer by copying input objects and parameters
     * of another backward layer
     * @param context   Context to manage backward layer
     * @param other     Backward layer to be used as the source to initialize the input objects
     *                  and parameters of the backward layer
     */
    public BackwardLayer(DaalContext context, BackwardLayer other) {
        super(context);
    }

    /**
     * Constructs the backward layer
     * @param context   Context to manage the backward layer
     */
    public BackwardLayer(DaalContext context) {
        super(context);
    }

    public BackwardLayer(DaalContext context, long cLayer) {
        super(context);
        cObject = cLayer;
    }

    /**
     * Returns the structure that contains result of the backward layer
     * \return Structure that contains result of the backward layer
     */
    public BackwardResult getLayerResult() {
        return new BackwardResult(getContext(), cGetResult(cObject));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * \return Structure that contains input object of the backward layer
     */
    public BackwardInput getLayerInput() {
        return new BackwardInput(getContext(), cGetInput(cObject));
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * \return Structure that contains parameters of the backward layer
     */
    public Parameter getLayerParameter() {
        return new Parameter(getContext(), cGetParameter(cObject));
    }

    /**
     * Returns the newly allocated backward layer with a copy of input objects
     * and parameters of this backward layer
     * @param context   Context to manage backward layer
     *
     * @return The newly allocated backward layer
     */
    @Override
    public BackwardLayer clone(DaalContext context) {
        return new BackwardLayer(context, this);
    }

    /**
     * Releases memory allocated for the backward layer of the neural network
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    private native long cGetParameter(long cObject);
    private native long cGetInput(long cObject);
    private native long cGetResult(long cObject);
    private native void cDispose(long cObject);
}
/** @} */
/** @} */
