/* file: LayerDescriptor.java */
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
 * @defgroup layers Layers
 * @brief Contains classes for neural network layers
 * @ingroup neural_networks
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.SerializableBase;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERDESCRIPTOR"></a>
 * \brief Class defining descriptor for layer on both forward and backward stages and its parameters
 */
public class LayerDescriptor extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the layer descriptor
     */
    public long cObject;

    /**
     * Constructs the descriptor for layer and its parameters
     * @param context   Context to manage the layer descriptor
     */
    public LayerDescriptor(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    /**
     * Constructs the descriptor for layer and its parameters
     * @param context    Context to manage the layer descriptor
     * @param index      Index of the layer in the network
     * @param layer      Layer algorithm
     * @param nextLayers Layers following the current layer in the network
     */
    public LayerDescriptor(DaalContext context, long index, LayerIface layer, NextLayers nextLayers) {
        super(context);
        cObject = cInit(index, layer.cObject, nextLayers.cObject);
    }

    public LayerDescriptor(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Releases memory allocated for the native layer descriptor object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    private native long cInit();
    private native long cInit(long index, long layerAddr, long nextLayersAddr);
    private native void cDispose(long cObject);
}
/** @} */
