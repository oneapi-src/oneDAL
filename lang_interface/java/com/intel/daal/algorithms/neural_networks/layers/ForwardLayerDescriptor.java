/* file: ForwardLayerDescriptor.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.SerializableBase;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARDLAYERDESCRIPTOR"></a>
 * \brief Class defining descriptor for layer on forward stage
 */
public class ForwardLayerDescriptor extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the layer descriptor
     */
    public long cObject;

    /**
     * Constructs the descriptor for layer
     * @param context   Context to manage the layer descriptor
     */
    public ForwardLayerDescriptor(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    /**
     * Constructs the descriptor for layer and its parameters
     * @param context    Context to manage the layer descriptor
     * @param index      Index of the layer in the network
     * @param layer      Forward layer algorithm
     * @param nextLayers Layers following the current layer in the network
     */
    public ForwardLayerDescriptor(DaalContext context, long index, ForwardLayer layer, NextLayers nextLayers) {
        super(context);
        cObject = cInit(index, layer.cObject, nextLayers.cObject);
    }

    public ForwardLayerDescriptor(DaalContext context, long cObject) {
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
