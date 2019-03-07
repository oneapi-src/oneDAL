/* file: Pooling1dKernelSize.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DKERNELSIZE"></a>
 * \brief Data structure representing the size of the 1D subtensor from which the element is computed
 */
public final class Pooling1dKernelSize {
    private long[] size; /*!< Array of sizes of the 1D subtensor from which the element is computed */

    /**
    * Constructs KernelSize with parameters
    * @param first  Size of the first dimension of the 1D subtensor
    */
    public Pooling1dKernelSize(long first) {
        size = new long[1];
        size[0] = first;
    }

    /**
     *  Sets the the array of sizes of the 1D subtensor from which the element is computed
    * @param first  Size of the first dimension of the 1D subtensor
     */
    public void setSize(long first) {
        size[0] = first;
    }

    /**
    *  Gets the array of sizes of the 1D subtensor from which the element is computed
    * @return Array of sizes of the 1D subtensor from which the element is computed
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
