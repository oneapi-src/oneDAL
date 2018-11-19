/* file: EluBatch.java */
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
 * @defgroup elu_layers Exponential Linear Unit (ELU) Layer
 * @brief Contains classes for the ELU layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the Exponential Linear Unit (ELU) layer
 */
package com.intel.daal.algorithms.neural_networks.layers.elu;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELU__ELUBATCH"></a>
 * @brief Provides methods for the Exponential Linear Unit (ELU) layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ELUFORWARD-ALGORITHM">Forward ELU layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-ELUBACKWARD-ALGORITHM">Backward ELU layer description and usage models</a> -->
 *
 * @par References
 *      - @ref EluForwardBatch class
 *      - @ref EluBackwardBatch class
 */
public class EluBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public  EluMethod        method;        /*!< Computation method for the layer */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the ELU layer
     * @param context    Context to manage the ELU layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref EluMethod
     */
    public EluBatch(DaalContext context, Class<? extends Number> cls, EluMethod method) {
        super(context);

        this.method = method;

        if (method != EluMethod.defaultDense) {
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

        forwardLayer = (ForwardLayer)(new EluForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new EluBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
