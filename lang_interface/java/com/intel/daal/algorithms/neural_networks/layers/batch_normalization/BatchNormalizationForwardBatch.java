/* file: BatchNormalizationForwardBatch.java */
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
 * @defgroup batch_normalization_forward_batch Batch
 * @ingroup batch_normalization_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward batch normalization layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-BATCH_NORMALIZATIONFORWARD-ALGORITHM">Forward batch normalization layer description and usage models</a> -->
 *
 * \par References
 *      - @ref BatchNormalizationLayerDataId class
 */
public class BatchNormalizationForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  BatchNormalizationForwardInput input;     /*!< %Input data */
    public  BatchNormalizationMethod       method;    /*!< Computation method for the layer */
    public  BatchNormalizationParameter    parameter; /*!< BatchNormalizationParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward batch normalization layer by copying input objects of another forward batch normalization layer
     * @param context    Context to manage the forward batch normalization layer
     * @param other      A forward batch normalization layer to be used as the source to
     *                   initialize the input objects of the forward batch normalization layer
     */
    public BatchNormalizationForwardBatch(DaalContext context, BatchNormalizationForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new BatchNormalizationForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new BatchNormalizationParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward batch normalization layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref BatchNormalizationMethod
     */
    public BatchNormalizationForwardBatch(DaalContext context, Class<? extends Number> cls, BatchNormalizationMethod method) {
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
        input = new BatchNormalizationForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new BatchNormalizationParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    BatchNormalizationForwardBatch(DaalContext context, Class<? extends Number> cls, BatchNormalizationMethod method, long cObject) {
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
        input = new BatchNormalizationForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new BatchNormalizationParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward batch normalization layer
     * @return  Forward batch normalization layer result
     */
    @Override
    public BatchNormalizationForwardResult compute() {
        super.compute();
        BatchNormalizationForwardResult result = new BatchNormalizationForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward batch normalization layer
     * @param result    Structure to store the result of the forward batch normalization layer
     */
    public void setResult(BatchNormalizationForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public BatchNormalizationForwardResult getLayerResult() {
        return new BatchNormalizationForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public BatchNormalizationForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public BatchNormalizationParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward batch normalization layer
     * with a copy of input objects of this forward batch normalization layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward batch normalization layer
     */
    @Override
    public BatchNormalizationForwardBatch clone(DaalContext context) {
        return new BatchNormalizationForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
