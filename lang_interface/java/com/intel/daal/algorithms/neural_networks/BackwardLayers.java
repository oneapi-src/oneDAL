/* file: BackwardLayers.java */
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
 * @ingroup neural_networks
 * @{
 */
/**
 * @brief Contains classes for for training and prediction using neural network
 */
package com.intel.daal.algorithms.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.SerializableBase;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__BACKWARDLAYERS"></a>
 * \brief Represents a collection of backward stages of neural network layers
 */
public class BackwardLayers extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the collection
     */
    public long cObject;

    /**
     * Constructs the collection of backward stages of neural network layers in the batch processing mode
     * @param context   Context to manage the collection
     */
    public BackwardLayers(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public BackwardLayers(DaalContext context, long cObject) {
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
     * Gets a backward layer with the given index from the collection
     * @param index Index of the backward layer
     */
    public BackwardLayer get(long index) {
        return new BackwardLayer(getContext(), cGet(cObject, index));
    }

    /**
     * Adds a backward layer to the end of the collection
     * @param layer Backward layer object
     */
    public void pushBack(BackwardLayer layer) {
        cPushBack(cObject, layer.cObject);
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
    private native void cPushBack(long cObject, long layerAddr);
    private native void cDispose(long cObject);
}
/** @} */
