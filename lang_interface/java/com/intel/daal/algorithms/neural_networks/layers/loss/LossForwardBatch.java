/* file: LossForwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward loss layer in the batch processing mode
 * \n <a href="DAAL-REF-LOSSFORWARD">Forward loss layer description and usage models</a>
 *
 * \par References
 *      - @ref LossForwardInput class
 *      - @ref LossForwardResult class
 */
public class LossForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the forward loss layer by copying input objects of another forward loss layer
     * @param context    Context to manage the forward loss layer
     * @param other      A forward loss layer to be used as the source to initialize the input objects of the forward loss layer
     */
    public LossForwardBatch(DaalContext context, LossForwardBatch other) {
        super(context);
    }

    public LossForwardBatch(DaalContext context) {
        super(context);
    }

    public LossForwardBatch(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Computes the result of the forward loss layer
     * @return  Forward loss layer result
     */
    @Override
    public LossForwardResult compute() {
        super.compute();
        LossForwardResult result = new LossForwardResult(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public LossForwardResult getLayerResult() {
        return new LossForwardResult(getContext(), cGetResult(cObject));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * \return Structure that contains input object of the forward layer
     */
    @Override
    public LossForwardInput getLayerInput() {
        return new LossForwardInput(getContext(), cGetInput(cObject));
    }
    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public LossParameter getLayerParameter() {
        return null;
    }

    /**
     * Returns the newly allocated forward loss layer
     * with a copy of input objects of this forward loss layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward loss layer
     */
    @Override
    public LossForwardBatch clone(DaalContext context) {
        return new LossForwardBatch(context, this);
    }

    private native long cGetInput(long cAlgorithm);
    private native long cGetResult(long cAlgorithm);
}
