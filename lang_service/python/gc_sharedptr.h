/* file: gc_sharedptr.h */
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

/*
//++
//  Implementation garbage collection hooks for numpy arrays passed to Intel(R) DAAL.
//--
*/

#ifndef __GC_SHAREDPTR_INCLUDED__
#define __GC_SHAREDPTR_INCLUDED__

#include <map>
#include <vector>
#include <services/daal_shared_ptr.h>

typedef std::vector< const PyObject * > gco_vec;
typedef std::map< const void*, gco_vec > gc_map;
static gc_map _pa_map;

static void increment_py_ref(const void *tbl, PyObject * obj) {
    if( ! obj ) return;
    SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    Py_INCREF( obj );
    _pa_map[tbl].push_back(obj);
    SWIG_PYTHON_THREAD_END_BLOCK;
}

static void decrement_py_ref(const void * tbl) {
    SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    gc_map::iterator it = _pa_map.find(tbl);
    if (it != _pa_map.end()) {
        for( gco_vec::iterator i = it->second.begin(); i != it->second.end(); ++i ) {
            Py_DECREF(*i);
        }
        _pa_map.erase(it);
    }
    SWIG_PYTHON_THREAD_END_BLOCK;
}

template<typename T>
static void decrement_py_ref_and_delete(const T * tbl) {
    decrement_py_ref(tbl);
    delete tbl;
}

#endif // __GC_SHAREDPTR_INCLUDED__
