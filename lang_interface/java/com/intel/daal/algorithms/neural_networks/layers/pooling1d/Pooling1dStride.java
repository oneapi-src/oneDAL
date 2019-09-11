/* file: Pooling1dStride.java */
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
 * @ingroup pooling1d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling1d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DSTRIDE"></a>
 * \brief Data structure representing the intervals on which the subtensors for one-dimensional pooling are computed
 */
public final class Pooling1dStride {
    private long[] size;     /*!< Array of intervals on which the subtensors for one-dimensional pooling are selected */

    /**
    * Constructs Stride with parameters
    * @param first   Interval over the first dimension on which the one-dimensional pooling is performed
    */
    public Pooling1dStride(long first) {
        size = new long[1];
        size[0] = first;
    }

    /**
     *  Sets the array of intervals on which the subtensors for one-dimensional pooling are selected
    * @param first   Interval over the first dimension on which the one-dimensional pooling is performed
     */
    public void setSize(long first) {
        size[0] = first;
    }

    /**
    *  Gets the array of intervals on which the subtensors for one-dimensional pooling are selected
    * @return Array of intervals on which the subtensors for one-dimensional pooling are selected
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
