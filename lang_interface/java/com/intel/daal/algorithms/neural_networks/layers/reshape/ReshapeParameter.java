/* file: ReshapeParameter.java */
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
 * @ingroup reshape
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.reshape;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RESHAPE__RESHAPEPARAMETER"></a>
 * \brief Class that specifies parameters of the reshape layer
 */
public class ReshapeParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     * Constructs the parameter of the reshape layer
     * @param context Context to manage the parameter of the reshape layer
     */
    public ReshapeParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }
    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage the parameter
     * @param cObject Address of C++ parameter
     */
    public ReshapeParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Gets the reshape dimensions
     * @return Array representing the reshape dimensions
     */
    public long[] getReshapeDimensions() {
        return cGetReshapeDimensions(cObject);
    }

    /**
     * Sets the array representing the reshape dimensions
     * @param dims   The array representing the reshape dimensions
     */
    public void setReshapeDimensions(long[] dims) {
        cSetReshapeDimensions(cObject, dims);
    }

    private native long cInit();
    private native void cSetReshapeDimensions(long cObject, long[] dims);
    private native long[] cGetReshapeDimensions(long cObject);
}
/** @} */
