/* file: LocallyConnected2dBackwardBatch.java */
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
 * @defgroup locallyconnected2d_backward_batch Batch
 * @ingroup locallyconnected2d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.locallyconnected2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__LOCALLYCONNECTED2DBACKWARDBATCH"></a>
 * \brief Class that computes the results of the 2D locally connected layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOCALLYCONNECTED2DBACKWAR">Backward 2D locally connected layer description and usage models</a> -->
 *
 * \par References
 *      - @ref LocallyConnected2dLayerDataId class
 */
public class LocallyConnected2dBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  LocallyConnected2dBackwardInput input;     /*!< %Input data */
    public  LocallyConnected2dMethod        method;    /*!< Computation method for the layer */
    public  LocallyConnected2dParameter     parameter; /*!< LocallyConnected2dParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward 2D locally connected layer by copying input objects of backward 2D locally connected layer
     * @param context    Context to manage the backward 2D locally connected layer
     * @param other      A backward 2D locally connected layer to be used as the source to initialize the input objects of the backward 2D locally connected layer
     */
    public LocallyConnected2dBackwardBatch(DaalContext context, LocallyConnected2dBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new LocallyConnected2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LocallyConnected2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward 2D locally connected layer
     * @param context    Context to manage the backward 2D locally connected layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LocallyConnected2dMethod
     */
    public LocallyConnected2dBackwardBatch(DaalContext context, Class<? extends Number> cls, LocallyConnected2dMethod method) {
        super(context);

        this.method = method;

        if (method != LocallyConnected2dMethod.defaultDense) {
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
        input = new LocallyConnected2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LocallyConnected2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    LocallyConnected2dBackwardBatch(DaalContext context, Class<? extends Number> cls, LocallyConnected2dMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != LocallyConnected2dMethod.defaultDense) {
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
        input = new LocallyConnected2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LocallyConnected2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward 2D locally connected layer
     * @return  Backward 2D locally connected layer result
     */
    @Override
    public LocallyConnected2dBackwardResult compute() {
        super.compute();
        LocallyConnected2dBackwardResult result = new LocallyConnected2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward 2D locally connected layer
     * @param result    Structure to store the result of the backward 2D locally connected layer
     */
    public void setResult(LocallyConnected2dBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public LocallyConnected2dBackwardResult getLayerResult() {
        return new LocallyConnected2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public LocallyConnected2dBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public LocallyConnected2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward 2D locally connected layer
     * with a copy of input objects of this backward 2D locally connected layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward 2D locally connected layer
     */
    @Override
    public LocallyConnected2dBackwardBatch clone(DaalContext context) {
        return new LocallyConnected2dBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
