/* file: Pooling3dKernelSizes.java */
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
 * @ingroup pooling3d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DKERNELSIZES"></a>
 * \brief Data structure representing the size of the three-dimensional subtensor
 */
public final class Pooling3dKernelSizes {
    private long[] size; /*!< Array of sizes of the three-dimensional kernel subtensor */

    /**
    * Constructs KernelSizes with parameters
    * @param first  Size of the first dimension of the three-dimensional subtensor
    * @param second Size of the second dimension of the three-dimensional subtensor
    * @param third  Size of the third dimension of the three-dimensional subtensor
    */
    public Pooling3dKernelSizes(long first, long second, long third) {
        size = new long[3];
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
     *  Sets the the array of sizes of the three-dimensional kernel subtensor
    * @param first  Size of the first dimension of the three-dimensional subtensor
    * @param second Size of the second dimension of the three-dimensional subtensor
    * @param third  Size of the third dimension of the three-dimensional subtensor
     */
    public void setSize(long first, long second, long third) {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Gets the array of sizes of the three-dimensional kernel subtensor
    * @return Array of sizes of the three-dimensional kernel subtensor
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
