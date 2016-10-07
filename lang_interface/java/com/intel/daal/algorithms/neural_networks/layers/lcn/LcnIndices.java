/* file: LcnIndices.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

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
