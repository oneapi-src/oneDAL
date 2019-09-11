/* file: ContextClient.java */
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
 * @defgroup memory Managing Memory
 * @brief Contains classes that implement memory allocation and deallocation.
 * @ingroup services
 * @{
 */
/**
 * @brief Contains classes that implement service functionality including memory management,
 *        information about environment, and library version information.
 */
package com.intel.daal.services;

/**
 *  <a name="DAAL-CLASS-SERVICES__CONTEXTCLIENT"></a>
 * @brief Class for management by deletion of the memory allocated for the native C++ object
 */
public abstract class ContextClient implements Disposable {
    private DaalContext _ownContext;

    /**
     * Default constructor
     */
    public ContextClient() {
        _ownContext = null;
    }

    /**
     * Constructs Client of Context and initializes it by provided Context
     * @param context   Context to manage the Client of Context
     */
    public ContextClient(DaalContext context) {
        context.add(this);
        _ownContext = context;
    }

    /**
     * Gets Context
     * @return Context
     */
    public DaalContext getContext() {
        return _ownContext;
    }

    /**
     * Changes Context
     */
    public void changeContext(DaalContext context) {
        if (_ownContext != null) {
            _ownContext.remove(this);
        }
        if (context != null) {
            context.add(this);
        }
        _ownContext = context;
    }

    @Override
    public abstract void dispose();
}
/** @} */
