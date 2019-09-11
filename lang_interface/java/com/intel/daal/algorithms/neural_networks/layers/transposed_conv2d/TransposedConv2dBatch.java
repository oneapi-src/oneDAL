/* file: TransposedConv2dBatch.java */
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
 * @defgroup transposed_conv2d Two-dimensional Transposed Convolution Layer
 * @brief Contains classes for neural network 2D transposed convolution layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the two-dimensional (2D) transposed convolution layer
 */
package com.intel.daal.algorithms.neural_networks.layers.transposed_conv2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TRANSPOSED_CONV2D__TRANSPOSEDCONV2DBATCH"></a>
 * @brief Provides methods for the 2D transposed convolution layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-TRANSPOSED_CONV2DFORWARD-ALGORITHM">Forward 2D transposed convolution layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-TRANSPOSED_CONV2DBACKWARD-ALGORITHM">Backward 2D transposed convolution layer description and usage models</a> -->
 *
 * @par References
 *      - @ref TransposedConv2dForwardBatch class
 *      - @ref TransposedConv2dBackwardBatch class
 */
public class TransposedConv2dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    TransposedConv2dMethod        method;      /*!< Computation method for the layer */
    public    TransposedConv2dParameter     parameter;   /*!< Convolution layer parameters */
    protected Precision                     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the 2D transposed convolution layer
     * @param context    Context to manage the 2D transposed convolution layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref TransposedConv2dMethod
     */
    public TransposedConv2dBatch(DaalContext context, Class<? extends Number> cls, TransposedConv2dMethod method) {
        super(context);

        this.method = method;

        if (method != TransposedConv2dMethod.defaultDense) {
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
        parameter = new TransposedConv2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new TransposedConv2dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new TransposedConv2dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
