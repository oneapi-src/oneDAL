/* file: LossBackwardBatch.java */
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
 * @defgroup loss_backward_batch Batch
 * @ingroup loss_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward loss layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOSSBACKWARD">Backward loss layer description and usage models</a> -->
 */
public class LossBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward loss layer by copying input objects of backward loss layer
     * @param context    Context to manage the backward loss layer
     * @param other      A backward loss layer to be used as the source to initialize the input objects of the backward loss layer
     */
    public LossBackwardBatch(DaalContext context, LossBackwardBatch other) {
        super(context);
    }

    /**
     * Constructs the backward loss layer
     * @param context Context to manage the backward loss layer
     */
    public LossBackwardBatch(DaalContext context) {
        super(context);
    }

    public LossBackwardBatch(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Computes the result of the backward loss layer
     * @return  Backward loss layer result
     */
    @Override
    public LossBackwardResult compute() {
        super.compute();
        LossBackwardResult result = new LossBackwardResult(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public LossBackwardResult getLayerResult() {
        return new LossBackwardResult(getContext(), cGetResult(cObject));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * \return Structure that contains input object of the backward layer
     */
    @Override
    public LossBackwardInput getLayerInput() {
        return new LossBackwardInput(getContext(), cGetInput(cObject));
    }
    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public LossParameter getLayerParameter() {
        return null;
    }

    /**
     * Returns the newly allocated backward loss layer
     * with a copy of input objects of this backward loss layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward loss layer
     */
    @Override
    public LossBackwardBatch clone(DaalContext context) {
        return new LossBackwardBatch(context, this);
    }

    private native long cGetInput(long cAlgorithm);
    private native long cGetResult(long cAlgorithm);
}
/** @} */
