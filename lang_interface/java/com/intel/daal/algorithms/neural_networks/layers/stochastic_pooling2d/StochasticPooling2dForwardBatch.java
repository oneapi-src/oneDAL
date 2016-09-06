/* file: StochasticPooling2dForwardBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

package com.intel.daal.algorithms.neural_networks.layers.stochastic_pooling2d;

import com.intel.daal.algorithms.neural_networks.layers.pooling2d.Pooling2dIndices;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__STOCHASTIC_POOLING2D__STOCHASTICPOOLING2DFORWARDBATCH"></a>
 * @brief Class that computes the results of the forward two-dimensional stochastic pooling layer in the batch processing mode
 * \n<a href="DAAL-REF-STOCHASTICPOOLING2DFORWARD-ALGORITHM">Forward two-dimensional stochastic pooling layer description and usage models</a>
 *
 * @par References
 *      - @ref StochasticPooling2dMethod class
 *      - @ref StochasticPooling2dLayerDataId class
 *      - @ref StochasticPooling2dForwardInput class
 *      - @ref StochasticPooling2dForwardResult class
 */
public class StochasticPooling2dForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  StochasticPooling2dForwardInput input;     /*!< %Input data */
    public  StochasticPooling2dMethod       method;    /*!< Computation method for the layer */
    public  StochasticPooling2dParameter    parameter; /*!< StochasticPooling2dParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the forward two-dimensional stochastic pooling layer by copying input objects of
     * another forward two-dimensional stochastic pooling layer
     * @param context    Context to manage the forward two-dimensional stochastic pooling layer
     * @param other      A forward two-dimensional stochastic pooling layer to be used as the source to
     *                   initialize the input objects of the forward two-dimensional stochastic pooling layer
     */
    public StochasticPooling2dForwardBatch(DaalContext context, StochasticPooling2dForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new StochasticPooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new StochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward two-dimensional stochastic pooling layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref StochasticPooling2dMethod
     * @param nDim       Number of dimensions in input data
     */
    public StochasticPooling2dForwardBatch(DaalContext context, Class<? extends Number> cls, StochasticPooling2dMethod method, long nDim) {
        super(context);

        this.method = method;

        if (method != StochasticPooling2dMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), nDim);
        input = new StochasticPooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new StochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    StochasticPooling2dForwardBatch(DaalContext context, Class<? extends Number> cls, StochasticPooling2dMethod method, long cObject, long nDim) {
        super(context);

        this.method = method;

        if (method != StochasticPooling2dMethod.defaultDense) {
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
        input = new StochasticPooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new StochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        Pooling2dIndices sd = new Pooling2dIndices(nDim - 2, nDim - 1);
        parameter.setIndices(sd);
    }

    /**
     * Computes the result of the forward two-dimensional stochastic pooling layer
     * @return  Forward two-dimensional stochastic pooling layer result
     */
    @Override
    public StochasticPooling2dForwardResult compute() {
        super.compute();
        StochasticPooling2dForwardResult result = new StochasticPooling2dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward two-dimensional stochastic pooling layer
     * @param result    Structure to store the result of the forward two-dimensional stochastic pooling layer
     */
    public void setResult(StochasticPooling2dForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public StochasticPooling2dForwardResult getLayerResult() {
        return new StochasticPooling2dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public StochasticPooling2dForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public StochasticPooling2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward two-dimensional stochastic pooling layer
     * with a copy of input objects of this forward two-dimensional stochastic pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward two-dimensional stochastic pooling layer
     */
    @Override
    public StochasticPooling2dForwardBatch clone(DaalContext context) {
        return new StochasticPooling2dForwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
