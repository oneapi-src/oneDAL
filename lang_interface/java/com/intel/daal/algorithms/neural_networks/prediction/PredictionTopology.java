/* file: PredictionTopology.java */
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
 * @ingroup neural_networks_prediction
 * @{
 */
package com.intel.daal.algorithms.neural_networks.prediction;

import com.intel.daal.algorithms.neural_networks.layers.ForwardLayerDescriptor;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__PREDICTIONTOPOLOGY"></a>
 * \brief Represents a collection of neural network forward layer descriptors
 */
public class PredictionTopology extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the collection
     */
    public long cObject;

    /**
     * Constructs the collection of neural network forward layer descriptors
     * @param context   Context to manage the collection
     */
    public PredictionTopology(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public PredictionTopology(DaalContext context, long cObject) {
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
     * Gets the forward layer descriptor with the given index from the collection
     * @param index Index of the layer descriptor
     * @return Layer descriptor
     */
    public ForwardLayerDescriptor get(long index) {
        return new ForwardLayerDescriptor(getContext(), cGet(cObject, index));
    }

    /**
     * Adds forward layer descriptor to the end of the collection
     * @param layer Forward layer descriptor object
     */
    public long pushBack(ForwardLayer layer) {
        return cPushBack(cObject, layer.cObject);
    }

    /**
     * Adds forward layer descriptor to the end of the collection
     * @param layer Forward layer descriptor object
     */
    public long add(ForwardLayer layer) {
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
    private native void cDispose(long cObject);
    private native void cAddNext(long cObject, long index, long next);
}
/** @} */
