/* file: BatchNormalizationBackwardBatch.java */
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
 * @defgroup batch_normalization_backward_batch Batch
 * @ingroup batch_normalization_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward batch normalization layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-BATCH_NORMALIZATIONBACKWARD-ALGORITHM">Backward batch normalization layer description and usage models</a> -->
 *
 * \par References
 *      - @ref BatchNormalizationLayerDataId class
 */
public class BatchNormalizationBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  BatchNormalizationBackwardInput input;     /*!< %Input data */
    public  BatchNormalizationMethod        method;    /*!< Computation method for the layer */
    public  BatchNormalizationParameter     parameter; /*!< BatchNormalizationParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward batch normalization layer by copying input objects of backward batch normalization layer
     * @param context    Context to manage the backward batch normalization layer
     * @param other      A backward batch normalization layer to be used as the source
     *                   to initialize the input objects of the backward batch normalization layer
     */
    public BatchNormalizationBackwardBatch(DaalContext context, BatchNormalizationBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new BatchNormalizationBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new BatchNormalizationParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward batch normalization layer
     * @param context    Context to manage the backward batch normalization layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref BatchNormalizationMethod
     */
    public BatchNormalizationBackwardBatch(DaalContext context, Class<? extends Number> cls, BatchNormalizationMethod method) {
        super(context);

        this.method = method;

        if (method != BatchNormalizationMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new BatchNormalizationBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new BatchNormalizationParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    BatchNormalizationBackwardBatch(DaalContext context, Class<? extends Number> cls, BatchNormalizationMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != BatchNormalizationMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cObject;
        input = new BatchNormalizationBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new BatchNormalizationParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward batch normalization layer
     * @return  Backward batch normalization layer result
     */
    @Override
    public BatchNormalizationBackwardResult compute() {
        super.compute();
        BatchNormalizationBackwardResult result = new BatchNormalizationBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward batch normalization layer
     * @param result    Structure to store the result of the backward batch normalization layer
     */
    public void setResult(BatchNormalizationBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public BatchNormalizationBackwardResult getLayerResult() {
        return new BatchNormalizationBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public BatchNormalizationBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public BatchNormalizationParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward batch normalization layer
     * with a copy of input objects of this backward batch normalization layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward batch normalization layer
     */
    @Override
    public BatchNormalizationBackwardBatch clone(DaalContext context) {
        return new BatchNormalizationBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
