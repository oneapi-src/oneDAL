/* file: FullyConnectedForwardBatch.java */
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
 * @defgroup fullyconnected_forward_batch Batch
 * @ingroup fullyconnected_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.fullyconnected;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__FULLYCONNECTEDFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward fully-connected layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-FULLYCONNECTEDFORWARD">Forward fully-connected layer description and usage models</a> -->
 *
 * \par References
 *      - @ref FullyConnectedLayerDataId class
 */
public class FullyConnectedForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  FullyConnectedForwardInput input;     /*!< %Input data */
    public  FullyConnectedMethod       method;    /*!< Computation method for the layer */
    public  FullyConnectedParameter    parameter; /*!< FullyConnectedParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward fully-connected layer by copying input objects of another forward fully-connected layer
     * @param context    Context to manage the forward fully-connected layer
     * @param other      A forward fully-connected layer to be used as the source to
     *                   initialize the input objects of the forward fully-connected layer
     */
    public FullyConnectedForwardBatch(DaalContext context, FullyConnectedForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new FullyConnectedForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new FullyConnectedParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward fully-connected layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref FullyConnectedMethod
     * @param nOutputs   A number of layer outputs
     */
    public FullyConnectedForwardBatch(DaalContext context, Class<? extends Number> cls, FullyConnectedMethod method, long nOutputs) {
        super(context);

        this.method = method;

        if (method != FullyConnectedMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), nOutputs);
        input = new FullyConnectedForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new FullyConnectedParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    FullyConnectedForwardBatch(DaalContext context, Class<? extends Number> cls, long cObject, FullyConnectedMethod method) {
        super(context);

        this.method = method;

        if (method != FullyConnectedMethod.defaultDense) {
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
        input = new FullyConnectedForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new FullyConnectedParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward fully-connected layer
     * @return  Forward fully-connected layer result
     */
    @Override
    public FullyConnectedForwardResult compute() {
        super.compute();
        FullyConnectedForwardResult result = new FullyConnectedForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward fully-connected layer
     * @param result    Structure to store the result of the forward fully-connected layer
     */
    public void setResult(FullyConnectedForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public FullyConnectedForwardResult getLayerResult() {
        return new FullyConnectedForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public FullyConnectedForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public FullyConnectedParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward fully-connected layer
     * with a copy of input objects of this forward fully-connected layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward fully-connected layer
     */
    @Override
    public FullyConnectedForwardBatch clone(DaalContext context) {
        return new FullyConnectedForwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long nOutputs);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
