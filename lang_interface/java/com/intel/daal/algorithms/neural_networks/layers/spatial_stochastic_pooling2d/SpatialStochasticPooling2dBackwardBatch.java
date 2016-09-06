/* file: SpatialStochasticPooling2dBackwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.spatial_stochastic_pooling2d;

import com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dIndices;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_STOCHASTIC_POOLING2D__SPATIALSTOCHASTICPOOLING2DBACKWARDBATCH"></a>
 * @brief Class that computes the results of the two-dimensional spatial stochastic pooling layer in the batch processing mode
 * \n<a href="DAAL-REF-STOCHASTICPOOLING2DBACKWARD-ALGORITHM">Backward two-dimensional spatial stochastic pooling layer description and usage models</a>
 *
 * @par References
 *      - @ref SpatialStochasticPooling2dMethod class
 *      - @ref SpatialStochasticPooling2dLayerDataId class
 *      - @ref SpatialStochasticPooling2dBackwardInput class
 *      - @ref SpatialStochasticPooling2dBackwardResult class
 */
public class SpatialStochasticPooling2dBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  SpatialStochasticPooling2dBackwardInput input;     /*!< %Input data */
    public  SpatialStochasticPooling2dMethod        method;    /*!< Computation method for the layer */
    public  SpatialStochasticPooling2dParameter     parameter; /*!< SpatialStochasticPooling2dParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward two-dimensional spatial stochastic pooling layer by copying input objects of backward two-dimensional spatial stochastic pooling layer
     * @param context    Context to manage the backward two-dimensional spatial stochastic pooling layer
     * @param other      A backward two-dimensional spatial stochastic pooling layer to be used as the source to initialize the input objects of
     *                   the backward two-dimensional spatial stochastic pooling layer
     */
    public SpatialStochasticPooling2dBackwardBatch(DaalContext context, SpatialStochasticPooling2dBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new SpatialStochasticPooling2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialStochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward two-dimensional spatial stochastic pooling layer
     * @param context       Context to manage the backward two-dimensional spatial stochastic pooling layer
     * @param cls           Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method        The layer computation method, @ref SpatialStochasticPooling2dMethod
     * @param pyramidHeight The value of pyramid height
     * @param nDim          Number of dimensions in input data
     */
    public SpatialStochasticPooling2dBackwardBatch(DaalContext context, Class<? extends Number> cls, SpatialStochasticPooling2dMethod method, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialStochasticPooling2dMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), pyramidHeight, nDim);
        input = new SpatialStochasticPooling2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialStochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    SpatialStochasticPooling2dBackwardBatch(DaalContext context, Class<? extends Number> cls, SpatialStochasticPooling2dMethod method, long cObject, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialStochasticPooling2dMethod.defaultDense) {
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
        input = new SpatialStochasticPooling2dBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialStochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        SpatialPooling2dIndices sd = new SpatialPooling2dIndices(nDim - 2, nDim - 1);
        parameter.setIndices(sd);
    }

    /**
     * Computes the result of the backward two-dimensional spatial stochastic pooling layer
     * @return  Backward two-dimensional spatial stochastic pooling layer result
     */
    @Override
    public SpatialStochasticPooling2dBackwardResult compute() {
        super.compute();
        SpatialStochasticPooling2dBackwardResult result = new SpatialStochasticPooling2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward two-dimensional spatial stochastic pooling layer
     * @param result    Structure to store the result of the backward two-dimensional spatial stochastic pooling layer
     */
    public void setResult(SpatialStochasticPooling2dBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public SpatialStochasticPooling2dBackwardResult getLayerResult() {
        return new SpatialStochasticPooling2dBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public SpatialStochasticPooling2dBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public SpatialStochasticPooling2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward two-dimensional spatial stochastic pooling layer
     * with a copy of input objects of this backward two-dimensional spatial stochastic pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward two-dimensional spatial stochastic pooling layer
     */
    @Override
    public SpatialStochasticPooling2dBackwardBatch clone(DaalContext context) {
        return new SpatialStochasticPooling2dBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long pyramidHeight, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
