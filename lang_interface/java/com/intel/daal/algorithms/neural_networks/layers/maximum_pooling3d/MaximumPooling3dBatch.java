/* file: MaximumPooling3dBatch.java */
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
 * @defgroup maximum_pooling3d Three-dimensional Max Pooling Layer
 * @brief Contains classes for maximum three-dimensional (3D) pooling layer
 * @ingroup pooling3d
 * @{
 */
/**
 * @brief Contains classes of the three-dimensional (3D) maximum pooling layer
 */
package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING3D__MAXIMUMPOOLING3DBATCH"></a>
 * @brief Provides methods for the three-dimensional maximum pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-MAXIMUMPOOLING3DFORWARD-ALGORITHM">Forward three-dimensional maximum pooling layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-MAXIMUMPOOLING3DBACKWARD-ALGORITHM">Backward three-dimensional maximum pooling layer description and usage models</a> -->
 *
 * @par References
 *      - @ref MaximumPooling3dForwardBatch class
 *      - @ref MaximumPooling3dBackwardBatch class
 */
public class MaximumPooling3dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    MaximumPooling3dMethod        method;      /*!< Computation method for the layer */
    public    MaximumPooling3dParameter     parameter;   /*!< Pooling layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the three-dimensional maximum pooling layer
     * @param context    Context to manage the three-dimensional maximum pooling layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref MaximumPooling3dMethod
     * @param nDim       Number of dimensions in input data
     */
    public MaximumPooling3dBatch(DaalContext context, Class<? extends Number> cls, MaximumPooling3dMethod method, long nDim) {
        super(context);

        this.method = method;

        if (method != MaximumPooling3dMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), nDim);
        parameter = new MaximumPooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new MaximumPooling3dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue()), nDim));
        backwardLayer = (BackwardLayer)(new MaximumPooling3dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(),
                                                                                                  method.getValue()), nDim));
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
