/* file: LrnForwardBatch.java */
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
 * @defgroup lrn_forward_batch Batch
 * @ingroup lrn_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lrn;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__LRNFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward local response normalization layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LRNFORWARD">Forward local response normalization layer description and usage models</a> -->
 *
 * \par References
 *      - @ref LrnLayerDataId class
 */
public class LrnForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  LrnForwardInput input;     /*!< %Input data */
    public  LrnMethod       method;    /*!< Computation method for the layer */
    public  LrnParameter    parameter; /*!< LrnParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward local response normalization layer by copying input objects of another forward local response normalization layer
     * @param context    Context to manage the forward local response normalization layer
     * @param other      A forward local response normalization layer to be used as the source to initialize the input objects
     *                   of the forward local response normalization layer
     */
    public LrnForwardBatch(DaalContext context, LrnForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new LrnForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LrnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward local response normalization layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LrnMethod
     */
    public LrnForwardBatch(DaalContext context, Class<? extends Number> cls, LrnMethod method) {
        super(context);

        this.method = method;

        if (method != LrnMethod.defaultDense) {
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
        input = new LrnForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LrnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward local response normalization layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LrnMethod
     */
    LrnForwardBatch(DaalContext context, Class<? extends Number> cls, LrnMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != LrnMethod.defaultDense) {
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
        input = new LrnForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LrnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward local response normalization layer
     * @return  Forward local response normalization layer result
     */
    @Override
    public LrnForwardResult compute() {
        super.compute();
        LrnForwardResult result = new LrnForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward local response normalization layer
     * @param result    Structure to store the result of the forward local response normalization layer
     */
    public void setResult(LrnForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public LrnForwardResult getLayerResult() {
        return new LrnForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public LrnForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public LrnParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward local response normalization layer
     * with a copy of input objects of this forward local response normalization layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward local response normalization layer
     */
    @Override
    public LrnForwardBatch clone(DaalContext context) {
        return new LrnForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
