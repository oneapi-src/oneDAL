/* file: SoftmaxBatch.java */
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
 * @defgroup softmax_layers Softmax Layer
 * @brief Contains classes of the softmax layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the softmax layer
 */
package com.intel.daal.algorithms.neural_networks.layers.softmax;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__SOFTMAXBATCH"></a>
 * @brief Provides methods for the softmax layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-SOFTMAXFORWARD-ALGORITHM">Forward softmax layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-SOFTMAXBACKWARD-ALGORITHM">Backward softmax layer description and usage models</a> -->
 *
 * @par References
 *      - @ref SoftmaxForwardBatch class
 *      - @ref SoftmaxBackwardBatch class
 */
public class SoftmaxBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    SoftmaxMethod        method;      /*!< Computation method for the layer */
    public    SoftmaxParameter     parameter;   /*!< Softmax layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the softmax layer
     * @param context    Context to manage the softmax layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref SoftmaxMethod
     */
    public SoftmaxBatch(DaalContext context, Class<? extends Number> cls, SoftmaxMethod method) {
        super(context);

        this.method = method;

        if (method != SoftmaxMethod.defaultDense) {
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
        parameter = new SoftmaxParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new SoftmaxForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new SoftmaxBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
