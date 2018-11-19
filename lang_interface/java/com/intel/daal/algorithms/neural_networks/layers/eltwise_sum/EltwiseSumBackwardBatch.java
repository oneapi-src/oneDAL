/* file: EltwiseSumBackwardBatch.java */
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
 * @defgroup eltwise_sum_backward_batch Batch
 * @ingroup eltwise_sum_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.eltwise_sum;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__ELTWISESUMBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward element-wise sum layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ELTWISESUMBACKWARD">Backward element-wise sum layer description and usage models</a> -->
 *
 * \par References
 *      - @ref EltwiseSumLayerDataId class
 */
public class EltwiseSumBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  EltwiseSumBackwardInput input;     /*!< %Input data */
    public  EltwiseSumMethod        method;    /*!< Computation method for the layer */
    public  EltwiseSumParameter     parameter; /*!< EltwiseSumParameters of the layer */
    private Precision               prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward element-wise sum layer by copying input objects of backward element-wise sum layer
     * @param context    Context to manage the backward element-wise sum layer
     * @param other      A backward element-wise sum layer to be used as the source
     *                   to initialize the input objects of the backward element-wise sum layer
     */
    public EltwiseSumBackwardBatch(DaalContext context, EltwiseSumBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new EltwiseSumBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new EltwiseSumParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward element-wise sum layer
     * @param context    Context to manage the backward element-wise sum layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref EltwiseSumMethod
     */
    public EltwiseSumBackwardBatch(DaalContext context, Class<? extends Number> cls, EltwiseSumMethod method) {
        super(context);

        this.method = method;

        if (method != EltwiseSumMethod.defaultDense) {
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
        input = new EltwiseSumBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new EltwiseSumParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    EltwiseSumBackwardBatch(DaalContext context, Class<? extends Number> cls, EltwiseSumMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != EltwiseSumMethod.defaultDense) {
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
        input = new EltwiseSumBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new EltwiseSumParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward element-wise sum layer
     * @return  Backward element-wise sum layer result
     */
    @Override
    public EltwiseSumBackwardResult compute() {
        super.compute();
        EltwiseSumBackwardResult result = new EltwiseSumBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward element-wise sum layer
     * @param result    Structure to store the result of the backward element-wise sum layer
     */
    public void setResult(EltwiseSumBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public EltwiseSumBackwardResult getLayerResult() {
        return new EltwiseSumBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public EltwiseSumBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public EltwiseSumParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward element-wise sum layer
     * with a copy of input objects of this backward element-wise sum layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward element-wise sum layer
     */
    @Override
    public EltwiseSumBackwardBatch clone(DaalContext context) {
        return new EltwiseSumBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
