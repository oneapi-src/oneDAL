/* file: Pooling2dStrides.java */
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
 * @ingroup pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DSTRIDES"></a>
 * \brief Data structure representing the intervals on which the subtensors for two-dimensional pooling are computed
 */
public final class Pooling2dStrides {
    private long[] size;     /*!< Array of intervals on which the subtensors for two-dimensional pooling are selected */

    /**
    * Constructs Strides with parameters
    * @param first   Interval over the first dimension on which the two-dimensional pooling is performed
    * @param second  Interval over the second dimension on which the two-dimensional pooling is performed
    */
    public Pooling2dStrides(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of intervals on which the subtensors for two-dimensional pooling are selected
    * @param first    Interval over the first dimension on which the two-dimensional pooling is performed
    * @param second   Interval over the second dimension on which the two-dimensional pooling is performed
     */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of intervals on which the subtensors for two-dimensional pooling are selected
    * @return Array of intervals on which the subtensors for two-dimensional pooling are selected
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
