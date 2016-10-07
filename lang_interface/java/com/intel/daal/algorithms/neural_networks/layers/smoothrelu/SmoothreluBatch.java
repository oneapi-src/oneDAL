/* file: SmoothreluBatch.java */
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
 * @brief Contains classes of the smooth rectified linear unit (smoothrelu) layer
 */
package com.intel.daal.algorithms.neural_networks.layers.smoothrelu;

import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SMOOTHRELU__SMOOTHRELUBATCH"></a>
 *  @brief Provides methods for the  smooth rectified linear unit (smoothrelu) layer in the batch processing mode
 * \n<a href="DAAL-REF-SMOOTHRELUFORWARD-ALGORITHM">Forward smoothrelu layer description and usage models</a>
 * \n<a href="DAAL-REF-SMOOTHRELUBACKWARD-ALGORITHM">Backward smoothrelu layer description and usage models</a>
 *
 * @par References
 *      - @ref SmoothreluForwardBatch class
 *      - @ref SmoothreluBackwardBatch class
 *      - @ref SmoothreluMethod class
 */
public class SmoothreluBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public  SmoothreluMethod        method;        /*!< Computation method for the layer */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the smoothrelu layer
     * @param context    Context to manage the smoothrelu layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref SmoothreluMethod
     */
    public SmoothreluBatch(DaalContext context, Class<? extends Number> cls, SmoothreluMethod method) {
        super(context);

        this.method = method;

        if (method != SmoothreluMethod.defaultDense) {
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

        forwardLayer = (ForwardLayer)(new SmoothreluForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new SmoothreluBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
