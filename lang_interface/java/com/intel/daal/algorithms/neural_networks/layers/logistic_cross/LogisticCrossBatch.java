/* file: LogisticCrossBatch.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @defgroup logistic_cross Logistic Cross-entropy Layer
 * @brief Contains classes for logistic cross-entropy layer
 * @ingroup loss
 * @{
 */
/**
 * @brief Contains classes of thelogistic cross-entropy layer
 */
package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC_CROSS__LOGISTICCROSSBATCH"></a>
 * @brief Provides methods for the logistic cross-entropy layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOGISTIC_CROSSFORWARD-ALGORITHM">Forward logistic cross-entropy layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-LOGISTIC_CROSSBACKWARD-ALGORITHM">Backward logistic cross-entropy layer description and usage models</a> -->
 *
 * @par References
 *      - @ref LogisticCrossForwardBatch class
 *      - @ref LogisticCrossBackwardBatch class
 */
public class LogisticCrossBatch extends com.intel.daal.algorithms.neural_networks.layers.loss.LossBatch {
    public  LogisticCrossMethod        method;        /*!< Computation method for the layer */
    public  LogisticCrossParameter     parameter;   /*!< Dropout layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the logistic cross-entropy layer
     * @param context    Context to manage the logistic cross-entropy layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LogisticCrossMethod
     */
    public LogisticCrossBatch(DaalContext context, Class<? extends Number> cls, LogisticCrossMethod method) {
        super(context);

        this.method = method;

        if (method != LogisticCrossMethod.defaultDense) {
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
        parameter = new LogisticCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new LogisticCrossForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new LogisticCrossBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
