/* file: ForwardLayer.java */
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
 * @defgroup layers_forward Forward Base Layer
 * @brief Contains classes for the forward stage of the neural network layer
 * @ingroup layers
 * @{
 */
/**
 * @defgroup layers_forward_batch Batch
 * @ingroup layers_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARDLAYER"></a>
 * \brief Class representing a forward layer of neural network
 */
public class ForwardLayer extends com.intel.daal.algorithms.AnalysisBatch {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward layer by copying input objects and parameters
     * of another forward layer
     * @param context   Context to manage forward layer
     * @param other     A forward layer to be used as the source to initialize the input objects
     *                  and parameters of the forward layer
     */
    public ForwardLayer(DaalContext context, ForwardLayer other) {
        super(context);
    }

    /**
     * Constructs the forward layer
     * @param context   Context to manage the forward layer
     */
    public ForwardLayer(DaalContext context) {
        super(context);
    }

    public ForwardLayer(DaalContext context, long cLayer) {
        super(context);
        cObject = cLayer;
    }

    /**
     * Returns the structure that contains result of the forward layer
     * \return Structure that contains result of the forward layer
     */
    public ForwardResult getLayerResult() {
        return new ForwardResult(getContext(), cGetResult(cObject));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * \return Structure that contains input object of the forward layer
     */
    public ForwardInput getLayerInput() {
        return new ForwardInput(getContext(), cGetInput(cObject));
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * \return Structure that contains parameters of the forward layer
     */
    public Parameter getLayerParameter() {
        return new Parameter(getContext(), cGetParameter(cObject));
    }

    /**
     * Returns the newly allocated forward layer with a copy of input objects
     * and parameters of this forward layer
     * @param context   Context to manage forward layer
     *
     * @return The newly allocated forward layer
     */
    @Override
    public ForwardLayer clone(DaalContext context) {
        return new ForwardLayer(context, this);
    }

    /**
     * Releases memory allocated for the forward layer of the neural network
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
