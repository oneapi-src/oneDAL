/* file: dbscan_cl_kernels.cl */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

/*
//++
//  Implementation of DBSCAN OpenCL kernels.
//--
*/

#ifndef __KNN_CL_KERNELS_CL__
#define __KNN_CL_KERNELS_CL__

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(dbscan_cl_kernels,

__kernel void point_distances(__global const algorithmFPType *data,
                             __global algorithmFPType *dist,
                             int row_id,
                             int power,
                             int dim,
                             int num_points) 
{

    const int index = get_global_id(0) * get_max_sub_group_size() + get_sub_group_id();
    if(index >= num_points)
        return;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    algorithmFPType sum = 0.0;
    for(int i = id; i < dim; i += size)
    {
        algorithmFPType val = fabs(data[row_id * dim + i] - data[index * dim + i]);
        algorithmFPType dm = 1.0;
        for(int j = 0; j < power; j++)
        {
            dm *= val;
        }
        sum += dm;
    }
    algorithmFPType ret = sub_group_reduce_add(sum);
    if(id == 0)
    {
        dist[index] = ret;
    }
}

__kernel void queue_block_distances(__global const algorithmFPType *data,
                             __global const int *queue,
                             __global  algorithmFPType *dist,
                             int queue_begin,
                             int queue_size,
                             int power,
                             int dim,
                             int num_points) 
{

    const int queue_pos = get_global_id(0);
    if(queue_pos >= queue_size)
        return;
    const int id = queue[queue_begin + queue_pos];
    const int group_num = get_num_sub_groups();
    const int group_id = get_sub_group_id();
    const int group_size = get_sub_group_size();
    const int local_id = get_sub_group_local_id();
    __global algorithmFPType * cur_dist = &dist[queue_pos * num_points];
    for(int j = group_id; j < num_points; j += group_num)
    {
        algorithmFPType sum = 0.0;
        for(int i = local_id; i < dim; i += group_size)
        {
            algorithmFPType val = fabs(data[id * dim + i] - data[j * dim + i]);
            algorithmFPType dm = 1.0;
            for(int j = 0; j < power; j++)
            {
                dm *= val;
            }
            sum += dm;
        }
        algorithmFPType ret = sub_group_reduce_add(sum);
        if(local_id == 0)
        {
            cur_dist[j] = ret;
        }
    }
}

__kernel void count_neighbors(__global const int *assignments,
                            __global const algorithmFPType *data,
                            int row_id,
                            int chunk_offset,
                            int chunk_size, 
                            int num_points, 
                            algorithmFPType eps,
                            __global const int *queue,
                             __global int *counters,
                             __global int *undefCounters    
                            ) 
{

    const int index = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int offset = index * chunk_size;
    const int number = num_points - offset < chunk_size ? num_points - offset : chunk_size;
    row_id = chunk_offset < 0 ? row_id : queue[row_id];
    const int distance_offset = offset + (chunk_offset < 0 ? 0 : chunk_offset);
    if(index >= num_points)
        return;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    __global const algorithmFPType * input = &data[distance_offset];
    __global const int * assigned = &assignments[offset];
    int count = 0;
    int new_count = 0;
    for(int i = id; i < number; i += size)
    {
        int nbr_flag = input[i] <= eps ? 1 : 0;
        int new_nbr_flag = nbr_flag > 0 && assigned[i] == _UNDEFINED_ && (i + offset != row_id) ? 1 : 0;
        count += nbr_flag;
        new_count += new_nbr_flag;
    }
    int ret_count = sub_group_reduce_add(count);
    int ret_new_count = sub_group_reduce_add(new_count);
    if(id == 0)
    {
        counters[index] = ret_count;
        undefCounters[index] = ret_new_count;
    }
}


__kernel void set_buffer_value(__global int *buffer,
                             int index,
                             int value)
{
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(1);
    if(local_id == 0 & global_id == 0)
        buffer[index] = value;
}
__kernel void set_buffer_value_by_queue_index(__global int *queue,
                            __global int *buffer,
                             int pos,
                             int value)
{
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(1);
    if(local_id == 0 & global_id == 0)
        buffer[queue[pos]] = value;
}

__kernel void count_offsets(__global const int * counters,
                            __global int * offsets,
                            int num_offsets)
{
    if(get_global_id(0) > 0 || get_sub_group_id() > 0)
        return;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    int start = 0;
    for(int i = id; i < num_offsets; i += size)
    {
        int cur_counter = counters[i];
        int local_offset = start + sub_group_scan_exclusive_add(cur_counter);
        int total_offset = sub_group_reduce_add(cur_counter);
        offsets[i] = local_offset;
        start += total_offset;        
    }
}

__kernel void push_to_queue(__global const algorithmFPType * distances,
                                    __global const int * offsets,
                                    __global int * assignments,
                                    __global int * queue,
                                    int queue_end,
                                    int row_id,
                                    int cluster_id,
                                    int chunk_offset,
                                    int chunk_size, 
                                    algorithmFPType eps,
                                    int num_points)
{
    const int index = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int offset = index * chunk_size;
    row_id = chunk_offset < 0 ? row_id : queue[row_id];
    const int dist_offset = offset + (chunk_offset < 0 ? 0 : chunk_offset);
    if(offset >= num_points)
        return;
    const int number = num_points - offset < chunk_size ? num_points - offset : chunk_size;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    __global const algorithmFPType * input = &distances[dist_offset];
    __global int * assigned = &assignments[offset];
    const int out_offset = offsets[index];
    int local_offset = 0;
    for(int i = id; i < number; i += size) {
        int nbr_flag = input[i] <= eps  ? 1 : 0;
        int new_nbr_flag = nbr_flag > 0 && assigned[i] == _UNDEFINED_ && (i + offset != row_id) ? 1 : 0;
        int pos = sub_group_scan_exclusive_add(new_nbr_flag);
        if(new_nbr_flag)
        {
            assigned[i] = cluster_id;
            queue[queue_end + out_offset + local_offset + pos] = offset + i;
        } 
        if(nbr_flag && (assigned[i] == _NOISE_))
        {
            assigned[i] = cluster_id;
        }
        local_offset += sub_group_reduce_add(new_nbr_flag);
    }
    if(index == 0 && id == 0)
    {
        assignments[row_id] = cluster_id;
    }
}

);

#endif
