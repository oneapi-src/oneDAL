/* file: ReluBatch.java */
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
 * @defgroup relu_layers Rectifier Linear Unit (ReLU) Layer
 * @brief Contains classes for the relu layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the rectified linear unit (relu) layer
 */
package com.intel.daal.algorithms.neural_networks.layers.relu;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RELU__RELUBATCH"></a>
 * @brief Provides methods for the rectified linear unit (relu) layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-RELUFORWARD-ALGORITHM">Forward relu layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-RELUBACKWARD-ALGORITHM">Backward relu layer description and usage models</a> -->
 *
 * @par References
 *      - @ref ReluForwardBatch class
 *      - @ref ReluBackwardBatch class
 */
public class ReluBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public  ReluMethod        method;        /*!< Computation method for the layer */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the relu layer
     * @param context    Context to manage the relu layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref ReluMethod
     */
    public ReluBatch(DaalContext context, Class<? extends Number> cls, ReluMethod method) {
        super(context);

        this.method = method;

        if (method != ReluMethod.defaultDense) {
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

        forwardLayer = (ForwardLayer)(new ReluForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new ReluBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
