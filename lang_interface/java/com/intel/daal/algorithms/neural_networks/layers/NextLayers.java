/* file: NextLayers.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__NEXTLAYERS"></a>
 * \brief Contains list of layer indices of layers following the current layer
 */
public class NextLayers extends ContextClient {
    /**
     * @brief Pointer to C++ implementation of the next layers list
     */
    public long cObject;

    /**
     * Constructs list of layer indices of layers following the current layer
     * @param context   Context to manage the list of layer indices
     */
    public NextLayers(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    /**
     * Constructs list of layer indices of layers following the current layer
     * @param context   Context to manage the list of layer indices
     * @param index1    First index of the next layer
     */
    public NextLayers(DaalContext context, long index1) {
        super(context);
        cObject = cInit(index1);
    }

    /**
     * Constructs list of layer indices of layers following the current layer
     * @param context   Context to manage the list of layer indices
     * @param index1    First index of the next layer
     * @param index2    Second index of the next layer
     */
    public NextLayers(DaalContext context, long index1, long index2) {
        super(context);
        cObject = cInit(index1, index2);
    }

    /**
     * Constructs list of layer indices of layers following the current layer
     * @param context   Context to manage the list of layer indices
     * @param index1    First index of the next layer
     * @param index2    Second index of the next layer
     * @param index3    Third index of the next layer
     */
    public NextLayers(DaalContext context, long index1, long index2, long index3) {
        super(context);
        cObject = cInit(index1, index2, index3);
    }

    /**
     * Constructs list of layer indices of layers following the current layer
     * @param context   Context to manage the list of layer indices
     * @param index1    First index of the next layer
     * @param index2    Second index of the next layer
     * @param index3    Third index of the next layer
     * @param index4    Fourth index of the next layer
     */
    public NextLayers(DaalContext context, long index1, long index2, long index3, long index4) {
        super(context);
        cObject = cInit(index1, index2, index3, index4);
    }

    /**
     * Constructs list of layer indices of layers following the current layer
     * @param context   Context to manage the list of layer indices
     * @param index1    First index of the next layer
     * @param index2    Second index of the next layer
     * @param index3    Third index of the next layer
     * @param index4    Fourth index of the next layer
     * @param index5    Fifth index of the next layer
     */
    public NextLayers(DaalContext context, long index1, long index2, long index3, long index4, long index5) {
        super(context);
        cObject = cInit(index1, index2, index3, index4, index5);
    }

    /**
     * Constructs list of layer indices of layers following the current layer
     * @param context   Context to manage the list of layer indices
     * @param index1    First index of the next layer
     * @param index2    Second index of the next layer
     * @param index3    Third index of the next layer
     * @param index4    Fourth index of the next layer
     * @param index5    Fifth index of the next layer
     * @param index6    Sixth index of the next layer
     */
    public NextLayers(DaalContext context, long index1, long index2, long index3, long index4, long index5, long index6) {
        super(context);
        cObject = cInit(index1, index2, index3, index4, index5, index6);
    }

    /**
     * Releases memory allocated for the native next layers list object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    private native long cInit();
    private native long cInit(long index1);
    private native long cInit(long index1, long index2);
    private native long cInit(long index1, long index2, long index3);
    private native long cInit(long index1, long index2, long index3, long index4);
    private native long cInit(long index1, long index2, long index3, long index4, long index5);
    private native long cInit(long index1, long index2, long index3, long index4, long index5, long index6);
    private native void cDispose(long cObject);
}
/** @} */
