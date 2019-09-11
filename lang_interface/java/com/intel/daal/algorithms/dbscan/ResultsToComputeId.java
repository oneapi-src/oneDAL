/* file: ResultsToComputeId.java */
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
 * @ingroup dbscan
 * @{
 */
package com.intel.daal.algorithms.dbscan;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__RESULTTOCOMPUTEID"></a>
 * \brief Available identifiers of results of the DBSCAN algorithm
 */
public final class ResultsToComputeId {

    public static final long none                    = 0x0000000000000000L; /*!< No optional result */
    public static final long computeCoreIndices      = 0x0000000000000001L; /*!< Compute table containing indices of core observations */
    public static final long computeCoreObservations = 0x0000000000000002L; /*!< Compute table containing core observations */
}/** @} */
