/* file: InputBatch.java */
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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__INPUTBATCH"></a>
 *  \brief %Base class to represent input arguments of the computation in the batch processing mode.
 *         Algorithm-specific input arguments are represented as derivative classes of the InputBatch class.
 */
public class InputBatch extends Input {
    /**
     * Constructs the input of the computation in batch processing mode
     * @param context Context to manage the input of the computation in the batch processing mode
     */
    public InputBatch(DaalContext context) {
        super(context);
    }
}
/** @} */
