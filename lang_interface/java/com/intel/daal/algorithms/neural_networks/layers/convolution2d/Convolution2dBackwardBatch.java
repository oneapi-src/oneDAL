/* file: Convolution2dBackwardBatch.java */
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
 * @defgroup convolution2d_backward_batch Batch
 * @ingroup convolution2d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.convolution2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__CONVOLUTION2DBACKWARDBATCH"></a>
 * \brief Class that computes the results of the 2D convolution layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-CONVOLUTION2DBACKWAR">Backward 2D convolution layer description and usage models</a> -->
 *
 * \par References
 *      - @ref Convolution2dLayerDataId class
 */
public class Convolution2dBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  Convolution2dBackwardInput input;     /*!< %Input data */
    public  Convolution2dMethod        method;    /*!< Computation method for the layer */
    public  Convolution2dParameter     parameter; /*!< Convolution2dParameters of the layer */
    private Precision                  prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward 2D convolution layer by copying input objects of backward 2D convolution layer
     * @param context    Context to manage the backward 2D convolution layer
     * @param other      A backward 2D convolution layer to be used as the source to initialize the input objects of the backward 2D convolution layer
     */
    public Convolution2dBackwardBatch(DaalContext context, Convolution2dBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Convolution2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Convolution2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward 2D convolution layer
     * @param context    Context to manage the backward 2D convolution layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref Convolution2dMethod
     */
    public Convolution2dBackwardBatch(DaalContext context, Class<? extends Number> cls, Convolution2dMethod method) {
        super(context);

        this.method = method;

        if (method != Convolution2dMethod.defaultDense) {
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
        input = new Convolution2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Convolution2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }


    /**
     * Constructs the backward 2D convolution layer
     * @param context    Context to manage the backward 2D convolution layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref Convolution2dMethod
     */
    Convolution2dBackwardBatch(DaalContext context, Class<? extends Number> cls, Convolution2dMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != Convolution2dMethod.defaultDense) {
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
        input = new Convolution2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Convolution2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward 2D convolution layer
     * @return  Backward 2D convolution layer result
     */
    @Override
    public Convolution2dBackwardResult compute() {
        super.compute();
        Convolution2dBackwardResult result = new Convolution2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward 2D convolution layer
     * @param result    Structure to store the result of the backward 2D convolution layer
     */
    public void setResult(Convolution2dBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public Convolution2dBackwardResult getLayerResult() {
        return new Convolution2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public Convolution2dBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public Convolution2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward 2D convolution layer
     * with a copy of input objects of this backward 2D convolution layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward 2D convolution layer
     */
    @Override
    public Convolution2dBackwardBatch clone(DaalContext context) {
        return new Convolution2dBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
