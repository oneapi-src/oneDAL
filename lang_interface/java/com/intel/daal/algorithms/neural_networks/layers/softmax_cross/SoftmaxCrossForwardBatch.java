/* file: SoftmaxCrossForwardBatch.java */
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
 * @defgroup softmax_cross_forward_batch Batch
 * @ingroup softmax_cross_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.softmax_cross;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX_CROSS__SOFTMAXCROSSFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward softmax cross-entropy layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-SOFTMAX_CROSSFORWARD-ALGORITHM">Forward softmax cross-entropy layer description and usage models</a> -->
 *
 * \par References
 *      - @ref SoftmaxCrossLayerDataId class
 */
public class SoftmaxCrossForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.loss.LossForwardBatch {
    public  SoftmaxCrossForwardInput input;     /*!< %Input data */
    public  SoftmaxCrossMethod       method;    /*!< Computation method for the layer */
    public  SoftmaxCrossParameter    parameter; /*!< SoftmaxCrossParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward softmax cross-entropy layer by copying input objects of another forward softmax cross-entropy layer
     * @param context    Context to manage the forward softmax cross-entropy layer
     * @param other      A forward softmax cross-entropy layer to be used as the source to initialize the input objects of the forward softmax cross-entropy layer
     */
    public SoftmaxCrossForwardBatch(DaalContext context, SoftmaxCrossForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new SoftmaxCrossForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SoftmaxCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward softmax cross-entropy layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref SoftmaxCrossMethod
     */
    public SoftmaxCrossForwardBatch(DaalContext context, Class<? extends Number> cls, SoftmaxCrossMethod method) {
        super(context);

        this.method = method;

        if (method != SoftmaxCrossMethod.defaultDense) {
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
        input = new SoftmaxCrossForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SoftmaxCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    SoftmaxCrossForwardBatch(DaalContext context, Class<? extends Number> cls, SoftmaxCrossMethod method, long cObject) {
        super(context, cObject);

        this.method = method;

        if (method != SoftmaxCrossMethod.defaultDense) {
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
        input = new SoftmaxCrossForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SoftmaxCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward softmax cross-entropy layer
     * @return  Forward softmax cross-entropy layer result
     */
    @Override
    public SoftmaxCrossForwardResult compute() {
        super.compute();
        SoftmaxCrossForwardResult result = new SoftmaxCrossForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward softmax cross-entropy layer
     * @param result    Structure to store the result of the forward softmax cross-entropy layer
     */
    public void setResult(SoftmaxCrossForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public SoftmaxCrossForwardResult getLayerResult() {
        return new SoftmaxCrossForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public SoftmaxCrossForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public SoftmaxCrossParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward softmax cross-entropy layer
     * with a copy of input objects of this forward softmax cross-entropy layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward softmax cross-entropy layer
     */
    @Override
    public SoftmaxCrossForwardBatch clone(DaalContext context) {
        return new SoftmaxCrossForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
