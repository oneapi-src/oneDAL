/* file: SpatialAveragePooling2dForwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.spatial_average_pooling2d;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dIndices;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__SPATIALAVERAGEPOOLING2DFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward two-dimensional spatial average pooling layer in the batch processing mode
 * \n<a href="DAAL-REF-AVERAGEPOOLING2DFORWARD-ALGORITHM">Forward two-dimensional spatial average pooling layer description and usage models</a>
 *
 * \par References
 *      - @ref SpatialAveragePooling2dMethod class
 *      - @ref SpatialAveragePooling2dLayerDataId class
 *      - @ref SpatialAveragePooling2dForwardInput class
 *      - @ref SpatialAveragePooling2dForwardResult class
 */
public class SpatialAveragePooling2dForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  SpatialAveragePooling2dForwardInput input;     /*!< %Input data */
    public  SpatialAveragePooling2dMethod       method;    /*!< Computation method for the layer */
    public  SpatialAveragePooling2dParameter    parameter; /*!< SpatialAveragePooling2dParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the forward two-dimensional spatial average pooling layer by copying input objects of
     * another forward two-dimensional spatial average pooling layer
     * @param context    Context to manage the forward two-dimensional spatial average pooling layer
     * @param other      A forward two-dimensional spatial average pooling layer to be used as the source
     *                   to initialize the input objects of the forward two-dimensional spatial average pooling layer
     */
    public SpatialAveragePooling2dForwardBatch(DaalContext context, SpatialAveragePooling2dForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new SpatialAveragePooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialAveragePooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward two-dimensional spatial average pooling layer
     * @param context       Context to manage the layer
     * @param cls           Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method        The layer computation method, @ref SpatialAveragePooling2dMethod
     * @param pyramidHeight The value of pyramid height
     * @param nDim          Number of dimensions in input data
     */
    public SpatialAveragePooling2dForwardBatch(DaalContext context, Class<? extends Number> cls, SpatialAveragePooling2dMethod method, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialAveragePooling2dMethod.defaultDense) {
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
        input = new SpatialAveragePooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialAveragePooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    SpatialAveragePooling2dForwardBatch(DaalContext context, Class<? extends Number> cls, SpatialAveragePooling2dMethod method, long cObject, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialAveragePooling2dMethod.defaultDense) {
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
        input = new SpatialAveragePooling2dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new SpatialAveragePooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        SpatialPooling2dIndices sd = new SpatialPooling2dIndices(nDim - 2, nDim - 1);
        parameter.setIndices(sd);
    }

    /**
     * Computes the result of the forward two-dimensional spatial average pooling layer
     * @return  Forward two-dimensional spatial average pooling layer result
     */
    @Override
    public SpatialAveragePooling2dForwardResult compute() {
        super.compute();
        SpatialAveragePooling2dForwardResult result = new SpatialAveragePooling2dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward two-dimensional spatial average pooling layer
     * @param result    Structure to store the result of the forward two-dimensional spatial average pooling layer
     */
    public void setResult(SpatialAveragePooling2dForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public SpatialAveragePooling2dForwardResult getLayerResult() {
        return new SpatialAveragePooling2dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public SpatialAveragePooling2dForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public SpatialAveragePooling2dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward two-dimensional spatial average pooling layer
     * with a copy of input objects of this forward two-dimensional spatial average pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward two-dimensional spatial average pooling layer
     */
    @Override
    public SpatialAveragePooling2dForwardBatch clone(DaalContext context) {
        return new SpatialAveragePooling2dForwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long pyramidHeight, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
