/* file: Convolution2dStride.java */
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
 * @ingroup convolution2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.convolution2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__CONVOLUTION2DSTRIDE"></a>
 * \brief Data structure representing the intervals on which the kernel should be applied to the input
 */
public final class Convolution2dStride {
    private long[] size;     /*!< Array of intervals on which the kernel should be applied to the input */

    /**
    * Constructs Convolution2dStride with parameters
    * @param first  The first interval on which the kernel should be applied to the input
    * @param second The second interval on which the kernel should be applied to the input
    */
    public Convolution2dStride(long first, long second) {
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
