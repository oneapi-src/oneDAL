/* file: TrainingTopology.java */
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

package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.algorithms.neural_networks.layers.LayerDescriptor;
import com.intel.daal.algorithms.neural_networks.layers.LayerIface;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.SerializableBase;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__TRAININGTOPOLOGY"></a>
 * \brief Represents a collection of neural network layer descriptors
 */
public class TrainingTopology extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the collection
     */
    public long cObject;

    /**
     * Constructs the collection of neural network layer descriptors
     * @param context   Context to manage the collection
     */
    public TrainingTopology(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public TrainingTopology(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Gets the size of the collection
     * @return Size of the collection
     */
    public long size() {
        return cSize(cObject);
    }

    /**
     * Gets the layer descriptor with the given index from the collection
     * @param index Index of the layer descriptor
     * @return Layer descriptor
     */
    public LayerDescriptor get(long index) {
        return new LayerDescriptor(getContext(), cGet(cObject, index));
    }

    /**
     * Adds a layer descriptor to the end of the collection
     * @param layer Layer descriptor object
     */
    public long pushBack(LayerIface layer) {
        return cPushBack(cObject, layer.cObject);
    }

    /**
     * Adds a layer descriptor to the end of the collection
     * @param layer Layer descriptor object
     */
    public long add(LayerIface layer) {
        return cPushBack(cObject, layer.cObject);
    }

    /**
     * Adds next layer to the given layer
     * @param index index of the layer to add next layer
     * @param next Index of the next layer
     */
    public void addNext(long index, long next) {
        cAddNext(cObject, index, next);
    }

    /**
     * Releases memory allocated for the native collection object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    private native long cInit();
    private native long cSize(long cObject);
    private native long cGet(long cObject, long index);
    private native long cPushBack(long cObject, long layerAddr);
    private native void cAddNext(long cObject, long index, long next);
    private native void cDispose(long cObject);
}
