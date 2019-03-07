/* file: AbsBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @ingroup abs_layers
 * @{
 */
/**
 * @brief Contains classes of the abs layer
 */
package com.intel.daal.algorithms.neural_networks.layers.abs;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ABS__ABSBATCH"></a>
 * @brief  Provides methods for the abs layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ABSFORWARD-ALGORITHM">Forward abs layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-ABSBACKWARD-ALGORITHM">Backward abs layer description and usage models</a> -->
 *
 * @par References
 *      - @ref AbsForwardBatch class
 *      - @ref AbsBackwardBatch class
 */
public class AbsBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public  AbsMethod        method;        /*!< Computation method for the layer */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the abs layer
     * @param context    Context to manage the abs layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref AbsMethod
     */
    public AbsBatch(DaalContext context, Class<? extends Number> cls, AbsMethod method) {
        super(context);

        this.method = method;

        if (method != AbsMethod.defaultDense) {
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

        forwardLayer = (ForwardLayer)(new AbsForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new AbsBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
