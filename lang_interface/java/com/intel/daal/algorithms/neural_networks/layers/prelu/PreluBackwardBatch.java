/* file: PreluBackwardBatch.java */
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
 * @defgroup prelu_backward_batch Batch
 * @ingroup prelu_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.prelu;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__PRELUBACKWARDBATCH"></a>
 * \brief Class that computes the results of the prelu layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-PRELUBACKWARD">Backward prelu layer description and usage models</a> -->
 *
 * \par References
 *      - @ref PreluLayerDataId class
 */
public class PreluBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  PreluBackwardInput input;     /*!< %Input data */
    public  PreluMethod        method;    /*!< Computation method for the layer */
    public  PreluParameter     parameter; /*!< PreluParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward prelu layer by copying input objects of backward prelu layer
     * @param context    Context to manage the backward prelu layer
     * @param other      A backward prelu layer to be used as the source to initialize the input objects of the backward prelu layer
     */
    public PreluBackwardBatch(DaalContext context, PreluBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new PreluBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new PreluParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward prelu layer
     * @param context    Context to manage the backward prelu layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref PreluMethod
     */
    public PreluBackwardBatch(DaalContext context, Class<? extends Number> cls, PreluMethod method) {
        super(context);

        this.method = method;

        if (method != PreluMethod.defaultDense) {
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
        input = new PreluBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new PreluParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }


    /**
     * Constructs the backward prelu layer
     * @param context    Context to manage the backward prelu layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref PreluMethod
     */
    PreluBackwardBatch(DaalContext context, Class<? extends Number> cls, PreluMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != PreluMethod.defaultDense) {
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
        input = new PreluBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new PreluParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward prelu layer
     * @return  Backward prelu layer result
     */
    @Override
    public PreluBackwardResult compute() {
        super.compute();
        PreluBackwardResult result = new PreluBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward prelu layer
     * @param result    Structure to store the result of the backward prelu layer
     */
    public void setResult(PreluBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public PreluBackwardResult getLayerResult() {
        return new PreluBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public PreluBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public PreluParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward prelu layer
     * with a copy of input objects of this backward prelu layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward prelu layer
     */
    @Override
    public PreluBackwardBatch clone(DaalContext context) {
        return new PreluBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
