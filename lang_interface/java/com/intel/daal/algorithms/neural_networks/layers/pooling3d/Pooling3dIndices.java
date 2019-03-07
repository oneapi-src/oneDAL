/* file: Pooling3dIndices.java */
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
 * @defgroup pooling3d Three-dimensional Pooling Layer
 * @brief Contains classes for the three-dimensional (3D) pooling layer
 * @ingroup layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DINDICES"></a>
 * \brief Data structure representing the dimension for convolution kernels
 */
public final class Pooling3dIndices {
    private long[] size;     /*!< Array of dimensions for convolution kernels */

    /**
    * Constructs Indices with parameters
    * @param first  The first dimension for convolution kernels
    * @param second  The second dimension for convolution kernels
    * @param third  The third dimension for convolution kernels
    */
    public Pooling3dIndices(long first, long second, long third) {
        size = new long[3];
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
     *  Sets the array of dimensions for convolution kernels
    * @param first  The first dimension for convolution kernels
    * @param second  The second dimension for convolution kernels
    * @param third  The third dimension for convolution kernels
    */
    public void setSize(long first, long second, long third) {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Gets the array of dimensions for convolution kernels
    * @return Array of dimensions for convolution kernels
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
