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

#ifndef _FGT_GRAPH_TRACE_IMPL_H
#define _FGT_GRAPH_TRACE_IMPL_H

#include "../tbb_profiling.h"

namespace tbb {
    namespace internal {

#if TBB_PREVIEW_FLOW_GRAPH_TRACE

static inline void fgt_internal_alias_input_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_INPUT_PORT, node, FLOW_NODE, name_index );
    itt_relation_add( ITT_DOMAIN_FLOW, node, FLOW_NODE, __itt_relation_is_parent_of, p, FLOW_INPUT_PORT );
}

static inline void fgt_internal_alias_output_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_OUTPUT_PORT, node, FLOW_NODE, name_index );
    itt_relation_add( ITT_DOMAIN_FLOW, node, FLOW_NODE, __itt_relation_is_parent_of, p, FLOW_OUTPUT_PORT );
}

template<typename InputType>
void alias_input_port(void *node, tbb::flow::receiver<InputType>* port, string_index name_index) {
    // TODO: Make fgt_internal_alias_input_port a function template?
    fgt_internal_alias_input_port( node, port, name_index);
}

template < typename PortsTuple, int N >
struct fgt_internal_input_alias_helper {
    static void alias_port( void *node, PortsTuple &ports ) {
        alias_input_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_INPUT_PORT_0 + N - 1) );
        fgt_internal_input_alias_helper<PortsTuple, N-1>::alias_port( node, ports );
    }
};

template < typename PortsTuple >
struct fgt_internal_input_alias_helper<PortsTuple, 0> {
    static void alias_port( void * /* node */, PortsTuple & /* ports */ ) { }
};

template<typename OutputType>
void alias_output_port(void *node, tbb::flow::sender<OutputType>* port, string_index name_index) {
    // TODO: Make fgt_internal_alias_output_port a function template?
    fgt_internal_alias_output_port( node, static_cast<void *>(port), name_index);
}

template < typename PortsTuple, int N >
struct fgt_internal_output_alias_helper {
    static void alias_port( void *node, PortsTuple &ports ) {
        alias_output_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_OUTPUT_PORT_0 + N - 1) );
        fgt_internal_output_alias_helper<PortsTuple, N-1>::alias_port( node, ports );
    }
};

template < typename PortsTuple >
struct fgt_internal_output_alias_helper<PortsTuple, 0> {
    static void alias_port( void * /*node*/, PortsTuple &/*ports*/ ) {
    }
};

static inline void fgt_internal_create_input_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_INPUT_PORT, node, FLOW_NODE, name_index );
}

static inline void fgt_internal_create_output_port( void *node, void *p, string_index name_index ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, p, FLOW_OUTPUT_PORT, node, FLOW_NODE, name_index );
}

template<typename InputType>
void register_input_port(void *node, tbb::flow::receiver<InputType>* port, string_index name_index) {
    // TODO: Make fgt_internal_create_input_port a function template?
    fgt_internal_create_input_port( node, port, name_index);
}

template < typename PortsTuple, int N >
struct fgt_internal_input_helper {
    static void register_port( void *node, PortsTuple &ports ) {
        register_input_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_INPUT_PORT_0 + N - 1) );
        fgt_internal_input_helper<PortsTuple, N-1>::register_port( node, ports );
    }
};

template < typename PortsTuple >
struct fgt_internal_input_helper<PortsTuple, 1> {
    static void register_port( void *node, PortsTuple &ports ) {
        register_input_port( node, &(tbb::flow::get<0>(ports)), FLOW_INPUT_PORT_0 );
    }
};

template<typename OutputType>
void register_output_port(void *node, tbb::flow::sender<OutputType>* port, string_index name_index) {
    // TODO: Make fgt_internal_create_output_port a function template?
    fgt_internal_create_output_port( node, static_cast<void *>(port), name_index);
}

template < typename PortsTuple, int N >
struct fgt_internal_output_helper {
    static void register_port( void *node, PortsTuple &ports ) {
        register_output_port( node, &(tbb::flow::get<N-1>(ports)), static_cast<tbb::internal::string_index>(FLOW_OUTPUT_PORT_0 + N - 1) );
        fgt_internal_output_helper<PortsTuple, N-1>::register_port( node, ports );
    }
};

template < typename PortsTuple >
struct fgt_internal_output_helper<PortsTuple,1> {
    static void register_port( void *node, PortsTuple &ports ) {
        register_output_port( node, &(tbb::flow::get<0>(ports)), FLOW_OUTPUT_PORT_0 );
    }
};

template< typename NodeType >
void fgt_multioutput_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  (void *)( static_cast< tbb::flow::receiver< typename NodeType::input_type > * >(const_cast< NodeType *>(node)) );
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc );
}

template< typename NodeType >
void fgt_multiinput_multioutput_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  const_cast<NodeType *>(node);
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc );
}

template< typename NodeType >
static inline void fgt_node_desc( const NodeType *node, const char *desc ) {
    void *addr =  (void *)( static_cast< tbb::flow::sender< typename NodeType::output_type > * >(const_cast< NodeType *>(node)) );
    itt_metadata_str_add( ITT_DOMAIN_FLOW, addr, FLOW_NODE, FLOW_OBJECT_NAME, desc );
}

static inline void fgt_graph_desc( void *g, const char *desc ) {
    itt_metadata_str_add( ITT_DOMAIN_FLOW, g, FLOW_GRAPH, FLOW_OBJECT_NAME, desc );
}

static inline void fgt_body( void *node, void *body ) {
    itt_relation_add( ITT_DOMAIN_FLOW, body, FLOW_BODY, __itt_relation_is_child_of, node, FLOW_NODE );
}

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node( string_index t, void *g, void *input_port, PortsTuple &ports ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, input_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_input_port( input_port, input_port, FLOW_INPUT_PORT_0 );
    fgt_internal_output_helper<PortsTuple, N>::register_port( input_port, ports );
}

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node_with_body( string_index t, void *g, void *input_port, PortsTuple &ports, void *body ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, input_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_input_port( input_port, input_port, FLOW_INPUT_PORT_0 );
    fgt_internal_output_helper<PortsTuple, N>::register_port( input_port, ports );
    fgt_body( input_port, body );
}

template< int N, typename PortsTuple >
static inline void fgt_multiinput_node( string_index t, void *g, PortsTuple &ports, void *output_port) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_output_port( output_port, output_port, FLOW_OUTPUT_PORT_0 );
    fgt_internal_input_helper<PortsTuple, N>::register_port( output_port, ports );
}

static inline void fgt_multiinput_multioutput_node( string_index t, void *n, void *g ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, n, FLOW_NODE, g, FLOW_GRAPH, t );
}

static inline void fgt_node( string_index t, void *g, void *output_port ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_output_port( output_port, output_port, FLOW_OUTPUT_PORT_0 );
}

static inline void fgt_node_with_body( string_index t, void *g, void *output_port, void *body ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, output_port, FLOW_NODE, g, FLOW_GRAPH, t );
    fgt_internal_create_output_port( output_port, output_port, FLOW_OUTPUT_PORT_0 );
    fgt_body( output_port, body );
}


static inline void fgt_node( string_index t, void *g, void *input_port, void *output_port ) {
    fgt_node( t, g, output_port );
    fgt_internal_create_input_port( output_port, input_port, FLOW_INPUT_PORT_0 );
}

static inline void fgt_node_with_body( string_index t, void *g, void *input_port, void *output_port, void *body ) {
    fgt_node_with_body( t, g, output_port, body );
    fgt_internal_create_input_port( output_port, input_port, FLOW_INPUT_PORT_0 );
}


static inline void  fgt_node( string_index t, void *g, void *input_port, void *decrement_port, void *output_port ) {
    fgt_node( t, g, input_port, output_port );
    fgt_internal_create_input_port( output_port, decrement_port, FLOW_INPUT_PORT_1 );
}

static inline void fgt_make_edge( void *output_port, void *input_port ) {
    itt_relation_add( ITT_DOMAIN_FLOW, output_port, FLOW_OUTPUT_PORT, __itt_relation_is_predecessor_to, input_port, FLOW_INPUT_PORT);
}

static inline void fgt_remove_edge( void *output_port, void *input_port ) {
    itt_relation_add( ITT_DOMAIN_FLOW, output_port, FLOW_OUTPUT_PORT, __itt_relation_is_sibling_of, input_port, FLOW_INPUT_PORT);
}

static inline void fgt_graph( void *g ) {
    itt_make_task_group( ITT_DOMAIN_FLOW, g, FLOW_GRAPH, NULL, FLOW_NULL, FLOW_GRAPH );
}

static inline void fgt_begin_body( void *body ) {
    itt_task_begin( ITT_DOMAIN_FLOW, body, FLOW_BODY, NULL, FLOW_NULL, FLOW_BODY );
}

static inline void fgt_end_body( void * ) {
    itt_task_end( ITT_DOMAIN_FLOW );
}

static inline void fgt_async_try_put_begin( void *node, void *port ) {
    itt_task_begin( ITT_DOMAIN_FLOW, port, FLOW_OUTPUT_PORT, node, FLOW_NODE, FLOW_OUTPUT_PORT );
}

static inline void fgt_async_try_put_end( void *, void * ) {
    itt_task_end( ITT_DOMAIN_FLOW );
}

static inline void fgt_async_reserve( void *node, void *graph ) {
    itt_region_begin( ITT_DOMAIN_FLOW, node, FLOW_NODE, graph, FLOW_GRAPH, FLOW_NULL );
}

static inline void fgt_async_commit( void *node, void *graph ) {
    itt_region_end( ITT_DOMAIN_FLOW, node, FLOW_NODE );
}

static inline void fgt_reserve_wait( void *graph ) {
    itt_region_begin( ITT_DOMAIN_FLOW, graph, FLOW_GRAPH, NULL, FLOW_NULL, FLOW_NULL );
}

static inline void fgt_release_wait( void *graph ) {
    itt_region_end( ITT_DOMAIN_FLOW, graph, FLOW_GRAPH );
}

#else // TBB_PREVIEW_FLOW_GRAPH_TRACE

static inline void fgt_graph( void * /*g*/ ) { }

template< typename NodeType >
static inline void fgt_multioutput_node_desc( const NodeType * /*node*/, const char * /*desc*/ ) { }

template< typename NodeType >
static inline void fgt_node_desc( const NodeType * /*node*/, const char * /*desc*/ ) { }

static inline void fgt_graph_desc( void * /*g*/, const char * /*desc*/ ) { }

static inline void fgt_body( void * /*node*/, void * /*body*/ ) { }

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node( string_index /*t*/, void * /*g*/, void * /*input_port*/, PortsTuple & /*ports*/ ) { }

template< int N, typename PortsTuple >
static inline void fgt_multioutput_node_with_body( string_index /*t*/, void * /*g*/, void * /*input_port*/, PortsTuple & /*ports*/, void * /*body*/ ) { }

template< int N, typename PortsTuple >
static inline void fgt_multiinput_node( string_index /*t*/, void * /*g*/, PortsTuple & /*ports*/, void * /*output_port*/ ) { }

static inline void fgt_multiinput_multioutput_node( string_index /*t*/, void * /*node*/, void * /*graph*/ ) { }

static inline void fgt_node( string_index /*t*/, void * /*g*/, void * /*output_port*/ ) { }
static inline void fgt_node( string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*output_port*/ ) { }
static inline void  fgt_node( string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*decrement_port*/, void * /*output_port*/ ) { }

static inline void fgt_node_with_body( string_index /*t*/, void * /*g*/, void * /*output_port*/, void * /*body*/ ) { }
static inline void fgt_node_with_body( string_index /*t*/, void * /*g*/, void * /*input_port*/, void * /*output_port*/, void * /*body*/ ) { }

static inline void fgt_make_edge( void * /*output_port*/, void * /*input_port*/ ) { }
static inline void fgt_remove_edge( void * /*output_port*/, void * /*input_port*/ ) { }

static inline void fgt_begin_body( void * /*body*/ ) { }
static inline void fgt_end_body( void *  /*body*/) { }

static inline void fgt_async_try_put_begin( void * /*node*/, void * /*port*/ ) { }
static inline void fgt_async_try_put_end( void * /*node*/ , void * /*port*/ ) { }
static inline void fgt_async_reserve( void * /*node*/, void * /*graph*/ ) { }
static inline void fgt_async_commit( void * /*node*/, void * /*graph*/ ) { }
static inline void fgt_reserve_wait( void * /*graph*/ ) { }
static inline void fgt_release_wait( void * /*graph*/ ) { }

#endif // TBB_PREVIEW_FLOW_GRAPH_TRACE

    } // namespace internal
} // namespace tbb

#endif
