/* file: Pooling2dIndices.java */
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
 * @defgroup pooling2d Two-dimensional Pooling Layer
 * @brief Contains classes for the two-dimensional (2D) pooling layer
 * @ingroup layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DINDICES"></a>
 * \brief Data structure representing the indices of the dimension on which two-dimensional pooling is performed
 */
public final class Pooling2dIndices {
    private long[] size;     /*!< Array of indices of the dimension on which two-dimensional pooling is performed */

    /**
    * Constructs Indices with parameters
    * @param first   The first dimension index
    * @param second  The second dimension index
    */
    public Pooling2dIndices(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of indices of the dimension on which two-dimensional pooling is performed
    * @param first   The first dimension index
    * @param second  The second dimension index
    */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of indices of the dimension on which two-dimensional pooling is performed
    * @return Array of indices of the dimension on which two-dimensional pooling is performed
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
