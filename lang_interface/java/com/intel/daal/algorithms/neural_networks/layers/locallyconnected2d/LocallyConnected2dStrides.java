/* file: LocallyConnected2dStrides.java */
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
 * @ingroup locallyconnected2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.locallyconnected2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__LOCALLYCONNECTED2DSTRIDE"></a>
 * \brief Data structure representing the intervals on which the kernel should be applied to the input
 */
public final class LocallyConnected2dStrides {
    private long[] size;     /*!< Array of intervals on which the kernel should be applied to the input */

    /**
    * Constructs LocallyConnected2dStrides with parameters
    * @param first  The first interval on which the kernel should be applied to the input
    * @param second The second interval on which the kernel should be applied to the input
    */
    public LocallyConnected2dStrides(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of intervals on which the kernel should be applied to the input
    * @param first  The first interval on which the kernel should be applied to the input
    * @param second The second interval on which the kernel should be applied to the input
     */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of intervals on which the kernel should be applied to the input
    * @return Array of intervals on which the kernel should be applied to the input
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
