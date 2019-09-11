/* file: Convolution2dForwardBatch.java */
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
 * @defgroup convolution2d_forward_batch Batch
 * @ingroup convolution2d_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.convolution2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__CONVOLUTION2DFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward 2D convolution layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-CONVOLUTION2DFORWARD">Forward 2D convolution layer description and usage models</a> -->
 *
 * \par References
 *      - @ref Convolution2dLayerDataId class
 */
public class Convolution2dForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  Convolution2dForwardInput input;     /*!< %Input data */
    public  Convolution2dMethod       method;    /*!< Computation method for the layer */
    public  Convolution2dParameter    parameter; /*!< Convolution2dParameters of the layer */
    private Precision                 prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward 2D convolution layer by copying input objects of another forward 2D convolution layer
     * @param context    Context to manage the forward 2D convolution layer
     * @param other      A forward 2D convolution layer to be used as the source to initialize the input objects of the forward 2D convolution layer
     */
    public Convolution2dForwardBatch(DaalContext context, Convolution2dForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Convolution2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Convolution2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward 2D convolution layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref Convolution2dMethod
     */
    public Convolution2dForwardBatch(DaalContext context, Class<? extends Number> cls, Convolution2dMethod method) {
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
        input = new Convolution2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Convolution2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward 2D convolution layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref Convolution2dMethod
     * @param cObject    Address of C++ forward batch
     */
    Convolution2dForwardBatch(DaalContext context, Class<? extends Number> cls, Convolution2dMethod method, long cObject) {
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
        input = new Convolution2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Convolution2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward 2D convolution layer
     * @return  Forward 2D convolution layer result
     */
    @Override
    public Convolution2dForwardResult compute() {
        super.compute();
        Convolution2dForwardResult result = new Convolution2dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward 2D convolution layer
     * @param result    Structure to store the result of the forward 2D convolution layer
     */
    public void setResult(Convolution2dForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public Convolution2dForwardResult getLayerResult() {
        return new Convolution2dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public Convolution2dForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public Convolution2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward 2D convolution layer
     * with a copy of input objects of this forward 2D convolution layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward 2D convolution layer
     */
    @Override
    public Convolution2dForwardBatch clone(DaalContext context) {
        return new Convolution2dForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
