/* file: ReshapeBackwardBatch.java */
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
 * @defgroup reshape_layers_backward_batch Batch
 * @ingroup reshape_layers_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.reshape;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.Parameter;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RESHAPE__RESHAPEBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward reshape layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-RESHAPEBACKWARD">Backward reshape layer description and usage models</a> -->
 *
 * \par References
 *      - @ref ReshapeLayerDataId class
 */
public class ReshapeBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  ReshapeBackwardInput input;    /*!< %Input data */
    public  ReshapeMethod        method;   /*!< Computation method for the layer */
    public  ReshapeParameter    parameter; /*!< Parameters of the layer */
    private Precision     prec;     /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward reshape layer by copying input objects of backward reshape layer
     * @param context    Context to manage the backward reshape layer
     * @param other      A backward reshape layer to be used as the source to initialize the input objects of the backward reshape layer
     */
    public ReshapeBackwardBatch(DaalContext context, ReshapeBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new ReshapeBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new ReshapeParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward reshape layer
     * @param context    Context to manage the backward reshape layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref ReshapeMethod
     */
    public ReshapeBackwardBatch(DaalContext context, Class<? extends Number> cls, ReshapeMethod method) {
        super(context);

        this.method = method;

        if (method != ReshapeMethod.defaultDense) {
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
        input = new ReshapeBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new ReshapeParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    ReshapeBackwardBatch(DaalContext context, Class<? extends Number> cls, ReshapeMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != ReshapeMethod.defaultDense) {
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
        input = new ReshapeBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new ReshapeParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward reshape layer
     * @return  Backward reshape layer result
     */
    @Override
    public ReshapeBackwardResult compute() {
        super.compute();
        ReshapeBackwardResult result = new ReshapeBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward reshape layer
     * @param result    Structure to store the result of the backward reshape layer
     */
    public void setResult(ReshapeBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public ReshapeBackwardResult getLayerResult() {
        return new ReshapeBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public ReshapeBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public Parameter getLayerParameter() {
        return null;
    }

    /**
     * Returns the newly allocated backward reshape layer
     * with a copy of input objects of this backward reshape layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward reshape layer
     */
    @Override
    public ReshapeBackwardBatch clone(DaalContext context) {
        return new ReshapeBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
