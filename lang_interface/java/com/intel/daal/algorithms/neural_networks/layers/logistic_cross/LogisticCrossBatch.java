/* file: LogisticCrossBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @defgroup logistic_cross Logistic Cross-entropy Layer
 * @brief Contains classes for logistic cross-entropy layer
 * @ingroup loss
 * @{
 */
/**
 * @brief Contains classes of thelogistic cross-entropy layer
 */
package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
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
