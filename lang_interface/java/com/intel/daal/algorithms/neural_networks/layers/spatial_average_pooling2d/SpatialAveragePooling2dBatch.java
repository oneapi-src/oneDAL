/* file: SpatialAveragePooling2dBatch.java */
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
 * @defgroup spatial_average_pooling2d Two-dimensional Spatial pyramid average Pooling Layer
 * @brief Contains classes for spatial pyramid average two-dimensional (2D) pooling layer
 * @ingroup spatial_pooling2d
 * @{
 */
/**
 * @brief Contains classes of the two-dimensional (2D) spatial average pooling layer
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_average_pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__SPATIALAVERAGEPOOLING2DBATCH"></a>
 * @brief Provides methods for the two-dimensional spatial average pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-AVERAGEPOOLING2DFORWARD-ALGORITHM">Forward two-dimensional spatial average pooling layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-AVERAGEPOOLING2DBACKWARD-ALGORITHM">Backward two-dimensional spatial average pooling layer description and usage models</a> -->
 *
 * @par References
 *      - @ref SpatialAveragePooling2dForwardBatch class
 *      - @ref SpatialAveragePooling2dBackwardBatch class
 */
public class SpatialAveragePooling2dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    SpatialAveragePooling2dMethod        method;      /*!< Computation method for the layer */
    public    SpatialAveragePooling2dParameter     parameter;   /*!< Pooling layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the two-dimensional spatial average pooling layer
     * @param context       Context to manage the two-dimensional spatial average pooling layer
     * @param cls           Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method        The layer computation method, @ref SpatialAveragePooling2dMethod
     * @param pyramidHeight The value of pyramid height
     * @param nDim          Number of dimensions in input data
     */
    public SpatialAveragePooling2dBatch(DaalContext context, Class<? extends Number> cls, SpatialAveragePooling2dMethod method, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialAveragePooling2dMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), pyramidHeight, nDim);
        parameter = new SpatialAveragePooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new SpatialAveragePooling2dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue()), pyramidHeight, nDim));
        backwardLayer = (BackwardLayer)(new SpatialAveragePooling2dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(),
                                                                                                  method.getValue()), pyramidHeight, nDim));
    }

    private native long cInit(int prec, int method, long pyramidHeight, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
