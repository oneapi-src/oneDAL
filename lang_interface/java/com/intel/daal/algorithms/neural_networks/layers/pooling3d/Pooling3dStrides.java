/* file: Pooling3dStrides.java */
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
 * @ingroup pooling3d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DSTRIDES"></a>
 * \brief Data structure representing the intervals on which the subtensors for pooling are selected
 */
public final class Pooling3dStrides {
    private long[] size;     /*!< Array of intervals on which the subtensors for pooling are selected */

    /**
    * Constructs Strides with parameters
    * @param first  The first interval on which the subtensors for pooling are selected
    * @param second The second interval on which the subtensors for pooling are selected
    * @param third The third interval on which the subtensors for pooling are selected
    */
    public Pooling3dStrides(long first, long second, long third) {
        size = new long[3];
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
     *  Sets the array of intervals on which the subtensors for pooling are selected
    * @param first The first interval on which the subtensors for pooling are selected
    * @param second The second interval on which the subtensors for pooling are selected
    * @param third The third interval on which the subtensors for pooling are selected
     */
    public void setSize(long first, long second, long third) {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }

    /**
    *  Gets the array of intervals on which the subtensors for pooling are selected
    * @return Array of intervals on which the subtensors for pooling are selected
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
