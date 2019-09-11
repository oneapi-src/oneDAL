/* file: LocallyConnected2dIndices.java */
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
 * @ingroup locallyconnected2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.locallyconnected2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__LOCALLYCONNECTED2DSPATIALDIMENSIONS"></a>
 * \brief Data structure representing the dimension for locally connected kernels
 */
public final class LocallyConnected2dIndices {
    private long[] size;     /*!< Array of dimensions for locally connected kernels */

    /**
    * Constructs LocallyConnected2dIndices with parameters
    * @param first  The first dimension for locally connected kernels
    * @param second The second dimension for locally connected kernels
    */
    public LocallyConnected2dIndices(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of dimensions for locally connected kernels
    * @param first  The first dimension for locally connected kernels
    * @param second The second dimension for locally connected kernels
    */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of dimensions for locally connected kernels
    * @return Array of dimensions for locally connected kernels
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
