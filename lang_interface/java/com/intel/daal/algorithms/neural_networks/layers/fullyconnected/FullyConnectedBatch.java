/* file: FullyConnectedBatch.java */
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
 * @brief Contains classes of the fully-connected layer
 */
package com.intel.daal.algorithms.neural_networks.layers.fullyconnected;

import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__FULLYCONNECTEDBATCH"></a>
 * @brief Provides methods for the fully-connected layer in the batch processing mode
 * \n<a href="DAAL-REF-FULLYCONNECTEDFORWARD-ALGORITHM">Forward fully-connected layer description and usage models</a>
 * \n<a href="DAAL-REF-FULLYCONNECTEDBACKWARD-ALGORITHM">Backward fully-connected layer description and usage models</a>
 *
 * @par References
 *      - @ref FullyConnectedForwardBatch class
 *      - @ref FullyConnectedBackwardBatch class
 *      - @ref FullyConnectedMethod class
 */
public class FullyConnectedBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public  FullyConnectedMethod        method;        /*!< Computation method for the layer */
    public    FullyConnectedParameter     parameter;   /*!< Fully-connected layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the fully-connected layer
     * @param context    Context to manage the fully-connected layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref FullyConnectedMethod
     * @param nOutputs   A number of layer outputs
     */
    public FullyConnectedBatch(DaalContext context, Class<? extends Number> cls, FullyConnectedMethod method, long nOutputs) {
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
        parameter = new FullyConnectedParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new FullyConnectedForwardBatch(context, cls, cGetForwardLayer(cObject, prec.getValue(), method.getValue()), method));
        backwardLayer = (BackwardLayer)(new FullyConnectedBackwardBatch(context, cls, cGetBackwardLayer(cObject, prec.getValue(), method.getValue()), method));
    }

    private native long cInit(int prec, int method, long nOutputs);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
