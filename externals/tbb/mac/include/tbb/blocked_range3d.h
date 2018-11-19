/*
    Copyright 2005-2017 Intel Corporation.

    The source code, information and material ("Material") contained herein is owned by
    Intel Corporation or its suppliers or licensors, and title to such Material remains
    with Intel Corporation or its suppliers or licensors. The Material contains
    proprietary information of Intel or its suppliers and licensors. The Material is
    protected by worldwide copyright laws and treaty provisions. No part of the Material
    may be used, copied, reproduced, modified, published, uploaded, posted, transmitted,
    distributed or disclosed in any way without Intel's prior express written permission.
    No license under any patent, copyright or other intellectual property rights in the
    Material is granted to or conferred upon you, either expressly, by implication,
    inducement, estoppel or otherwise. Any license under such intellectual property
    rights must be express and approved by Intel in writing.

    Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
    or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
    in any way.
*/

#ifndef __TBB_blocked_range3d_H
#define __TBB_blocked_range3d_H

#include "tbb_stddef.h"
#include "blocked_range.h"

namespace tbb {

//! A 3-dimensional range that models the Range concept.
/** @ingroup algorithms */
template<typename PageValue, typename RowValue=PageValue, typename ColValue=RowValue>
class blocked_range3d {
public:
    //! Type for size of an iteration range
    typedef blocked_range<PageValue> page_range_type;
    typedef blocked_range<RowValue>  row_range_type;
    typedef blocked_range<ColValue>  col_range_type;

private:
    page_range_type my_pages;
    row_range_type  my_rows;
    col_range_type  my_cols;

public:

    blocked_range3d( PageValue page_begin, PageValue page_end,
                     RowValue  row_begin,  RowValue row_end,
                     ColValue  col_begin,  ColValue col_end ) :
        my_pages(page_begin,page_end),
        my_rows(row_begin,row_end),
        my_cols(col_begin,col_end)
    {}

    blocked_range3d( PageValue page_begin, PageValue page_end, typename page_range_type::size_type page_grainsize,
                     RowValue  row_begin,  RowValue row_end,   typename row_range_type::size_type row_grainsize,
                     ColValue  col_begin,  ColValue col_end,   typename col_range_type::size_type col_grainsize ) :
        my_pages(page_begin,page_end,page_grainsize),
        my_rows(row_begin,row_end,row_grainsize),
        my_cols(col_begin,col_end,col_grainsize)
    {}

    //! True if range is empty
    bool empty() const {
        // Yes, it is a logical OR here, not AND.
        return my_pages.empty() || my_rows.empty() || my_cols.empty();
    }

    //! True if range is divisible into two pieces.
    bool is_divisible() const {
        return  my_pages.is_divisible() || my_rows.is_divisible() || my_cols.is_divisible();
    }

    blocked_range3d( blocked_range3d& r, split ) :
        my_pages(r.my_pages),
        my_rows(r.my_rows),
        my_cols(r.my_cols)
    {
        split split_obj;
        do_split(r, split_obj);
    }

#if __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES
    //! Static field to support proportional split
    static const bool is_splittable_in_proportion = true;

    blocked_range3d( blocked_range3d& r, proportional_split& proportion ) :
        my_pages(r.my_pages),
        my_rows(r.my_rows),
        my_cols(r.my_cols)
    {
        do_split(r, proportion);
    }
#endif /* __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES */

    //! The pages of the iteration space
    const page_range_type& pages() const {return my_pages;}

    //! The rows of the iteration space
    const row_range_type& rows() const {return my_rows;}

    //! The columns of the iteration space
    const col_range_type& cols() const {return my_cols;}

private:

    template <typename Split>
    void do_split( blocked_range3d& r, Split& split_obj)
    {
        if ( my_pages.size()*double(my_rows.grainsize()) < my_rows.size()*double(my_pages.grainsize()) ) {
            if ( my_rows.size()*double(my_cols.grainsize()) < my_cols.size()*double(my_rows.grainsize()) ) {
                my_cols.my_begin = col_range_type::do_split(r.my_cols, split_obj);
            } else {
                my_rows.my_begin = row_range_type::do_split(r.my_rows, split_obj);
            }
        } else {
            if ( my_pages.size()*double(my_cols.grainsize()) < my_cols.size()*double(my_pages.grainsize()) ) {
                my_cols.my_begin = col_range_type::do_split(r.my_cols, split_obj);
            } else {
                my_pages.my_begin = page_range_type::do_split(r.my_pages, split_obj);
            }
        }
    }
};

} // namespace tbb

#endif /* __TBB_blocked_range3d_H */
