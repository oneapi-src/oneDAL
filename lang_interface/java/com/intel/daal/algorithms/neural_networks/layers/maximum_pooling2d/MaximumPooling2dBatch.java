/* file: MaximumPooling2dBatch.java */
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
 * @brief Contains classes of the two-dimensional (2D) maximum pooling layer
 */
package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling2d;

import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING2D__MAXIMUMPOOLING2DBATCH"></a>
 * @brief Provides methods for the two-dimensional maximum pooling layer in the batch processing mode
 * \n<a href="DAAL-REF-MAXIMUMPOOLING2DFORWARD-ALGORITHM">Forward two-dimensional maximum pooling layer description and usage models</a>
 * \n<a href="DAAL-REF-MAXIMUMPOOLING2DBACKWARD-ALGORITHM">Backward two-dimensional maximum pooling layer description and usage models</a>
 *
 * @par References
 *      - @ref MaximumPooling2dForwardBatch class
 *      - @ref MaximumPooling2dBackwardBatch class
 *      - @ref MaximumPooling2dMethod class
 */
public class MaximumPooling2dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    MaximumPooling2dMethod        method;      /*!< Computation method for the layer */
    public    MaximumPooling2dParameter     parameter;   /*!< Pooling layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the two-dimensional maximum pooling layer
     * @param context    Context to manage the two-dimensional maximum pooling layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref MaximumPooling2dMethod
     * @param nDim       Number of dimensions in input data
     */
    public MaximumPooling2dBatch(DaalContext context, Class<? extends Number> cls, MaximumPooling2dMethod method, long nDim) {
        super(context);

        this.method = method;

        if (method != MaximumPooling2dMethod.defaultDense) {
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
        parameter = new MaximumPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new MaximumPooling2dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue()), nDim));
        backwardLayer = (BackwardLayer)(new MaximumPooling2dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(),
                                                                                                  method.getValue()), nDim));
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
