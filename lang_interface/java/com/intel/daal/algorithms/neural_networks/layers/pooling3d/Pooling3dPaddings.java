/* file: Pooling3dPaddings.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DPADDINGS"></a>
 * \brief Data structure representing the number of data elements to implicitly add to each size of
 *        the three-dimensional subtensor on which pooling is performed
 */
public final class Pooling3dPaddings {
    private long[] size;     /*!< Array of numbers of data elements to implicitly add to each size of
                                  the three-dimensional subtensor on which pooling is performed */
    /**
    * Constructs Paddings with parameters
    * @param first  The first number of data elements to implicitly add to each size of
    *               the three-dimensional subtensor on which pooling is performed
    * @param second The second number of data elements to implicitly add to each size of
    *               the three-dimensional subtensor on which pooling is performed
    * @param third The third number of data elements to implicitly add to each size of
    *               the three-dimensional subtensor on which pooling is performed
    */
    public Pooling3dPaddings(long first, long second, long third) {
        size = new long[3];
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Sets the array of numbers of data elements to implicitly add to each size of
    *  the three-dimensional subtensor on which pooling is performed
    * @param first  The first number of data elements to implicitly add to each size of
    *        the three-dimensional subtensor on which pooling is performed
    * @param second The second number of data elements to implicitly add to each size of
    *        the three-dimensional subtensor on which pooling is performed
    * @param third The third number of data elements to implicitly add to each size of
    *        the three-dimensional subtensor on which pooling is performed
    */
    public void setSize(long first, long second, long third) {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Gets the array of numbers of data elements to implicitly add to each size of
    *  the three-dimensional subtensor on which pooling is performed
    * @return Array of numbers of data elements to implicitly add to each size of
    *         he three-dimensional subtensor on which pooling is performed
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
