/* file: LcnIndices.java */
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
 * @ingroup lcn_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lcn;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNINDICES"></a>
 * \brief Data structure representing the indices of the two dimensions on which local contrast normalization is performed
 */
public final class LcnIndices {
    private long[] dims;     /*!< Array of indices of the two dimensions on which local contrast normalization is performed */

    /**
    * Constructs LcnIndices with parameters
    * @param first  The first index of the two dimensions on which local contrast normalization is performed
    * @param second The second index of the two dimensions on which local contrast normalization is performed
    */
    public LcnIndices(long first, long second) {
        dims = new long[2];
        dims[0] = first;
        dims[1] = second;
    }

    /**
     *  Sets the array of indices of the two dimensions on which local contrast normalization is performed
    * @param first  The first index of the two dimensions on which local contrast normalization is performed
    * @param second The second index of the two dimensions on which local contrast normalization is performed
    */
    public void setSize(long first, long second) {
        dims[0] = first;
        dims[1] = second;
    }

    /**
    *  Gets the array of indices of the two dimensions on which local contrast normalization is performed
    * @return Array of indices of the two dimensions on which local contrast normalization is performed
    */
    public long[] getSize() {
        return dims;
    }
}
/** @} */
