/* file: LossBatch.java */
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
 * @defgroup loss Loss Layer
 * @brief Contains classes for loss layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the loss layer
 */
package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSBATCH"></a>
 * @brief Provides methods for the loss layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOSSFORWARD-ALGORITHM">Forward loss layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-LOSSBACKWARD-ALGORITHM">Backward loss layer description and usage models</a> -->
 *
 * @par References
 *      - @ref LossForwardBatch class
 *      - @ref LossBackwardBatch class
 */
public class LossBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the loss layer
     * @param context    Context to manage the loss layer
     * @param cObject    Address of C++ object
     */
    public LossBatch(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
    * Constructs the loss layer
    * @param context    Context to manage the loss layer
    */
    public LossBatch(DaalContext context) {
        super(context);
    }
}
/** @} */
