/* file: Pooling2dKernelSizes.java */
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
 * @ingroup pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DKERNELSIZES"></a>
 * \brief Data structure representing the size of the 2D subtensor from which the element is computed
 */
public final class Pooling2dKernelSizes {
    private long[] size; /*!< Array of sizes of the 2D subtensor from which the element is computed */

    /**
    * Constructs KernelSizes with parameters
    * @param first  Size of the first dimension of the 2D subtensor
    * @param second  Size of the second dimension of the 2D subtensor
    */
    public Pooling2dKernelSizes(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Sets the the array of sizes of the 2D subtensor from which the element is computed
    * @param first  Size of the first dimension of the 2D subtensor
    * @param second  Size of the second dimension of the 2D subtensor
    */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of sizes of the 2D subtensor from which the element is computed
    * @return Array of sizes of the 2D subtensor from which the element is computed
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
