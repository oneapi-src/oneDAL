/* file: NextLayersCollection.java */
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
 * @defgroup neural_networks Neural Networks
 * @brief Contains classes for training and prediction using neural network
 * @ingroup training_and_prediction
 * @{
 */
package com.intel.daal.algorithms.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.NextLayers;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.SerializableBase;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__NEXTLAYERSCOLLECTION"></a>
 * \brief Represents a collection of neural network NextLayers objects
 */
public class NextLayersCollection extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the collection
     */
    public long cObject;

    /**
     * Constructs the collection of neural network NextLayers objects
     * @param context   Context to manage the collection
     */
    public NextLayersCollection(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public NextLayersCollection(DaalContext context, long cObject) {
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
     * Adds a NextLayers object to the end of the collection
     * @param nextLayers nextLayers object
     */
    public void pushBack(NextLayers nextLayers) {
        cPushBack(cObject, nextLayers.cObject);
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
    private native void cPushBack(long cObject, long nextLayersAddr);
    private native void cDispose(long cObject);
}
/** @} */
