/* file: SpatialStochasticPooling2dBatch.java */
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

/**
 * @brief Contains classes of the two-dimensional (2D) spatial stochastic pooling layer
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_stochastic_pooling2d;

import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_STOCHASTIC_POOLING2D__SPATIALSTOCHASTICPOOLING2DBATCH"></a>
 * @brief Provides methods for the two-dimensional spatial stochastic pooling layer in the batch processing mode
 * \n<a href="DAAL-REF-STOCHASTICPOOLING2DFORWARD-ALGORITHM">Forward two-dimensional spatial stochastic pooling layer description and usage models</a>
 * \n<a href="DAAL-REF-STOCHASTICPOOLING2DBACKWARD-ALGORITHM">Backward two-dimensional spatial stochastic pooling layer description and usage models</a>
 *
 * @par References
 *      - @ref SpatialStochasticPooling2dForwardBatch class
 *      - @ref SpatialStochasticPooling2dBackwardBatch class
 *      - @ref SpatialStochasticPooling2dMethod class
 */
public class SpatialStochasticPooling2dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    SpatialStochasticPooling2dMethod        method;      /*!< Computation method for the layer */
    public    SpatialStochasticPooling2dParameter     parameter;   /*!< Pooling layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the two-dimensional spatial stochastic pooling layer
     * @param context        Context to manage the two-dimensional spatial stochastic pooling layer
     * @param cls            Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method         The layer computation method, @ref SpatialStochasticPooling2dMethod
     * @param pyramidHeight  The value of pyramid height
     * @param nDim           Number of dimensions in input data
     */
    public SpatialStochasticPooling2dBatch(DaalContext context, Class<? extends Number> cls, SpatialStochasticPooling2dMethod method, long pyramidHeight, long nDim) {
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
        parameter = new SpatialStochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new SpatialStochasticPooling2dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue()), pyramidHeight, nDim));
        backwardLayer = (BackwardLayer)(new SpatialStochasticPooling2dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(),
                                                                                                  method.getValue()), pyramidHeight, nDim));
    }

    private native long cInit(int prec, int method, long pyramidHeight, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
