/* file: BatchNormalizationBatch.java */
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
 * @ingroup batch_normalization
 * @{
 */
/**
 * @brief Contains classes of the batch normalization layer
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONBATCH"></a>
 * @brief Provides methods for the batch normalization layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-BATCH_NORMALIZATIONFORWARD-ALGORITHM">Forward batch normalization layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-BATCH_NORMALIZATIONBACKWARD-ALGORITHM">Backward batch normalization layer description and usage models</a> -->
 *
 * @par References
 *      - @ref BatchNormalizationForwardBatch class
 *      - @ref BatchNormalizationBackwardBatch class
 */
public class BatchNormalizationBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    BatchNormalizationMethod        method;        /*!< Computation method for the layer */
    public    BatchNormalizationParameter     parameter;   /*!< BatchNormalizationBatch normalization layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the batch normalization layer
     * @param context    Context to manage the batch normalization layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref BatchNormalizationMethod
     */
    public BatchNormalizationBatch(DaalContext context, Class<? extends Number> cls, BatchNormalizationMethod method) {
        super(context);

        this.method = method;

        if (method != BatchNormalizationMethod.defaultDense) {
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
        parameter = new BatchNormalizationParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new BatchNormalizationForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new BatchNormalizationBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
