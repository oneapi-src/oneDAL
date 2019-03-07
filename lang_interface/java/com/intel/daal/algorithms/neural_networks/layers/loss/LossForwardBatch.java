/* file: LossForwardBatch.java */
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
 * @defgroup loss_forward_batch Batch
 * @ingroup loss_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward loss layer in the batch processing mode
 * <!-- \n <a href="DAAL-REF-LOSSFORWARD">Forward loss layer description and usage models</a> -->
 */
public class LossForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward loss layer by copying input objects of another forward loss layer
     * @param context    Context to manage the forward loss layer
     * @param other      A forward loss layer to be used as the source to initialize the input objects of the forward loss layer
     */
    public LossForwardBatch(DaalContext context, LossForwardBatch other) {
        super(context);
    }

    /**
     * Constructs the forward loss layer
     * @param context Context to manage the forward loss layer
     */
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
/** @} */
