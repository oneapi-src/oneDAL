/* file: BackwardLayer.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARDLAYER"></a>
 * \brief Class representing a backward layer of neural network
 */
public class BackwardLayer extends com.intel.daal.algorithms.AnalysisBatch {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
        return null;
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

    private native long cGetInput(long cObject);
    private native long cGetResult(long cObject);
    private native void cDispose(long cObject);
}
