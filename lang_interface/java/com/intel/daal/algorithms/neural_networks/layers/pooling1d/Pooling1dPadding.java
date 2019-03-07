/* file: Pooling1dPadding.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DPADDING"></a>
 * \brief Data structure representing the number of data elements to implicitly add
 *        to each side of the 1D subtensor on which one-dimensional pooling is performed
 */
public final class Pooling1dPadding {
    private long[] size;     /*!< Array of numbers of data elements to implicitly add to each size of
                                  the 1D subtensor on which one-dimensional pooling is performed */
    /**
    * Constructs Padding with parameters
    * @param first  Number of data elements to add to the the first dimension of the 1D subtensor
    */
    public Pooling1dPadding(long first) {
        size = new long[1];
        size[0] = first;
    }

    /**
    *  Sets the array of numbers of data elements to implicitly add to each size of
    *  the one-dimensional subtensor on which one-dimensional pooling is performed
    * @param first  Number of data elements to add to the the first dimension of the 1D subtensor
    */
    public void setSize(long first) {
        size[0] = first;
    }

    /**
    *  Gets the array of numbers of data elements to implicitly add to each size of
    *  the one-dimensional subtensor on which one-dimensional pooling is performed
    * @return Array of numbers of data elements to implicitly add to each size of
    *         he one-dimensional subtensor on which one-dimensional pooling is performed
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
