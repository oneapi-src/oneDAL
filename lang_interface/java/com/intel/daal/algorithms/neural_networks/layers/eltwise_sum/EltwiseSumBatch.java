/* file: EltwiseSumBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @defgroup eltwise_sum Element-wise Sum Layer
 * @brief Contains classes for the element-wise sum layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the element-wise sum layer
 */
package com.intel.daal.algorithms.neural_networks.layers.eltwise_sum;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__ELTWISESUMBATCH"></a>
 * @brief Provides methods for the element-wise sum layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ELTWISESUMFORWARD-ALGORITHM">Forward element-wise sum layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-ELTWISESUMBACKWARD-ALGORITHM">Backward element-wise sum layer description and usage models</a> -->
 *
 * @par References
 *      - @ref EltwiseSumForwardBatch class
 *      - @ref EltwiseSumBackwardBatch class
 */
public class EltwiseSumBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    EltwiseSumMethod        method;      /*!< Computation method for the layer */
    public    EltwiseSumParameter     parameter;   /*!< Element-wise sum layer parameters */
    protected Precision               prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the element-wise sum layer
     * @param context    Context to manage the element-wise sum layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref EltwiseSumMethod
     */
    public EltwiseSumBatch(DaalContext context, Class<? extends Number> cls, EltwiseSumMethod method) {
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

        parameter = new EltwiseSumParameter(context,
            cInitParameter(cObject, prec.getValue(), method.getValue()) );

        forwardLayer = (ForwardLayer)(new EltwiseSumForwardBatch(context, cls, method,
            cGetForwardLayer(cObject, prec.getValue(), method.getValue()) ));

        backwardLayer = (BackwardLayer)(new EltwiseSumBackwardBatch(context, cls, method,
            cGetBackwardLayer(cObject, prec.getValue(), method.getValue()) ));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
