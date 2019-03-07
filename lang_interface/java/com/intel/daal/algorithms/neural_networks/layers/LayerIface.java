/* file: LayerIface.java */
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
 * @ingroup layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.SerializableBase;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERIFACE"></a>
 * \brief Abstract class that specifies the interface of layer
 */
public abstract class LayerIface extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the layer
     */
    public long cObject;

    public  ForwardLayer  forwardLayer;  /*!< Forward stage of the layer algorithm */
    public  BackwardLayer backwardLayer; /*!< Backward stage of the layer algorithm */

    protected LayerIface(DaalContext context) {
        super(context);
    }

    /**
     * Releases memory allocated for the layer of the neural network
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    private native void cDispose(long cObject);
}
/** @} */
