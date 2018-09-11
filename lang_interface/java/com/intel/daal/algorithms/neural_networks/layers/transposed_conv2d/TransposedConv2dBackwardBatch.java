/* file: TransposedConv2dBackwardBatch.java */
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
 * @defgroup transposed_conv2d_backward_batch Batch
 * @ingroup transposed_conv2d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.transposed_conv2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TRANSPOSED_CONV2D__TRANSPOSEDCONV2DBACKWARDBATCH"></a>
 * \brief Class that computes the results of the 2D transposed convolution layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-TRANSPOSED_CONV2DBACKWARD-ALGORITHM">Backward 2D transposed convolution layer description and usage models</a> -->
 *
 * \par References
 *      - @ref TransposedConv2dLayerDataId class
 */
public class TransposedConv2dBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  TransposedConv2dBackwardInput input;     /*!< %Input data */
    public  TransposedConv2dMethod        method;    /*!< Computation method for the layer */
    public  TransposedConv2dParameter     parameter; /*!< TransposedConv2dParameters of the layer */
    private Precision                     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward 2D transposed convolution layer by copying input objects of backward 2D transposed convolution layer
     * @param context    Context to manage the backward 2D transposed convolution layer
     * @param other      A backward 2D transposed convolution layer to be used as the source to initialize the input objects of the backward 2D transposed convolution layer
     */
    public TransposedConv2dBackwardBatch(DaalContext context, TransposedConv2dBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new TransposedConv2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TransposedConv2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward 2D transposed convolution layer
     * @param context    Context to manage the backward 2D transposed convolution layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref TransposedConv2dMethod
     */
    public TransposedConv2dBackwardBatch(DaalContext context, Class<? extends Number> cls, TransposedConv2dMethod method) {
        super(context);

        this.method = method;

        if (method != TransposedConv2dMethod.defaultDense) {
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
        input = new TransposedConv2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TransposedConv2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }


    /**
     * Constructs the backward 2D transposed convolution layer
     * @param context    Context to manage the backward 2D transposed convolution layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref TransposedConv2dMethod
     */
    TransposedConv2dBackwardBatch(DaalContext context, Class<? extends Number> cls, TransposedConv2dMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != TransposedConv2dMethod.defaultDense) {
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
        input = new TransposedConv2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TransposedConv2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward 2D transposed convolution layer
     * @return  Backward 2D transposed convolution layer result
     */
    @Override
    public TransposedConv2dBackwardResult compute() {
        super.compute();
        TransposedConv2dBackwardResult result = new TransposedConv2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward 2D transposed convolution layer
     * @param result    Structure to store the result of the backward 2D transposed convolution layer
     */
    public void setResult(TransposedConv2dBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public TransposedConv2dBackwardResult getLayerResult() {
        return new TransposedConv2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public TransposedConv2dBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public TransposedConv2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward 2D transposed convolution layer
     * with a copy of input objects of this backward 2D transposed convolution layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward 2D transposed convolution layer
     */
    @Override
    public TransposedConv2dBackwardBatch clone(DaalContext context) {
        return new TransposedConv2dBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
