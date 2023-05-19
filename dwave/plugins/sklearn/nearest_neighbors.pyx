# distutils: language = c++
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "math.h":
    double log(double x) nogil

def calculate_mi(np.ndarray[np.float_t,ndim = 2] X,
                np.ndarray[np.float_t,ndim = 1] y,
                unsigned long refinement_factor = 5): 
        
    cdef unsigned long n_obs = X.shape[0]
    cdef float log_n_obs = log(n_obs)

    cdef unsigned long n_variables = X.shape[1]

    if(n_obs != y.shape[0]):
        raise ValueError("y is the wrong shape")

    cdef long sub_index_size = refinement_factor*int(np.round(log(n_obs))) + 2
    cdef total_state_space_size = pow(sub_index_size,3)

    cdef np.ndarray[np.float_t, ndim = 2] X_and_y = np.zeros((n_obs, n_variables + 1), dtype = float)
    
    X_and_y[:,:-1] = X
    X_and_y[:,-1] = y

    cdef np.ndarray[unsigned long, ndim = 2] assignments = assign_nn(X_and_y, sub_index_size)
    cdef np.ndarray[unsigned long, ndim = 4] assignment_counts_x = np.zeros((sub_index_size, sub_index_size, n_variables,n_variables), dtype=np.uint)
    cdef np.ndarray[unsigned long, ndim = 2] assignment_counts_x_univariate = np.zeros((sub_index_size, n_variables), dtype=np.uint)
    cdef np.ndarray[unsigned long, ndim = 1] assignment_counts_y = np.zeros((sub_index_size), dtype=np.uint)
    cdef np.ndarray[unsigned long, ndim = 5] assignment_counts_joint = np.zeros((sub_index_size, sub_index_size, sub_index_size, n_variables, n_variables), dtype=np.uint)
    cdef np.ndarray[unsigned long, ndim = 3] assignment_counts_joint_univariate = np.zeros((sub_index_size, sub_index_size, n_variables), dtype=np.uint)

    cdef np.ndarray[double, ndim = 2] mi_matrix = np.zeros((n_variables, n_variables))

    for variable_1 in range(n_variables):
        for variable_2 in range(n_variables):
            if variable_1 > variable_2:
                for obs in range(n_obs):
                    assignment_counts_x[assignments[obs, variable_1],assignments[obs, variable_2],variable_1, variable_2] += 1
                    assignment_counts_joint[assignments[obs, variable_1],assignments[obs, variable_2], assignments[obs, -1],variable_1, variable_2] += 1

        for obs in range(n_obs):
            assignment_counts_x_univariate[assignments[obs, variable_1], variable_1] += 1
            assignment_counts_joint_univariate[assignments[obs, variable_1], assignments[obs, -1], variable_1] += 1

    for obs in range(n_obs):
        assignment_counts_y[assignments[obs, -1]] += 1
    
    for variable_1 in range(n_variables):
        for variable_2 in range(n_variables):
            if variable_1 > variable_2:
                for i in range(sub_index_size): 
                    for j in range(sub_index_size): 
                        for k in range(sub_index_size):
                            if assignment_counts_joint[i,j,k,variable_1, variable_2] > 0:
                                ijk_term = (<float> assignment_counts_joint[i,j,k,variable_1, variable_2])
                                ijk_term *= log(ijk_term) - log(assignment_counts_x[i,j,variable_1, variable_2]) - log(assignment_counts_y[k]) + log_n_obs
                                mi_matrix[variable_1, variable_2] += ijk_term
                                mi_matrix[variable_2, variable_1] += ijk_term
        for i in range(sub_index_size): 
            for k in range(sub_index_size):
                if assignment_counts_joint_univariate[i,k,variable_1] > 0:
                    ik_term = (<float> assignment_counts_joint_univariate[i,k,variable_1])
                    ik_term *= log(ik_term) - log(assignment_counts_x_univariate[i,variable_1]) - log(assignment_counts_y[k]) + log_n_obs
                    mi_matrix[variable_1, variable_1] += ik_term
    
    return mi_matrix/n_obs

cdef assign_nn(np.ndarray[np.float_t,ndim = 2] X,
                long sub_index_size, 
                short seed = 12121):

    cdef unsigned long n_obs = X.shape[0]
    cdef unsigned long n_variables = X.shape[1]
    
    rng = np.random.default_rng(seed)

    cdef long[::1] full_index = np.zeros(n_obs, dtype=long)
    
    for i in range(n_obs): 
        full_index[i] = i

    cdef long[::1] sub_index = rng.choice(full_index, sub_index_size)

    # numpy floats are c doubles
    cdef np.ndarray[double, ndim=2] sort_values_x = X[sub_index,:].astype(float)

    cdef np.ndarray[unsigned long,ndim = 2] X_out = np.zeros((n_obs, n_variables), dtype=np.uint)

    cdef unsigned long start_index = (<unsigned long>  sub_index_size) >> 1

    #print("start index: ", start_index)
    for i in range(sort_values_x.shape[1]): 
        sort_values_x[:,i] = np.sort(sort_values_x[:,i])

        for obs in range(X.shape[0]):
            X_out[obs, i] = query(X[obs, i], sort_values_x[:,i],start_index)

    return X_out

cdef unsigned long query(double query_number,np.ndarray[double, ndim=1] query_list, unsigned long idx): 

    if query_list.shape[0] == 1: 
        return idx
    cdef unsigned long new_len = (<unsigned long> query_list.shape[0]) >> 1 
    cdef double pivot_value = query_list[new_len]
    cdef unsigned long new_idx_unit = (<unsigned long> new_len) >> 1 
    if query_number < pivot_value:
        #print("query : ", query_number, " pivot: ", pivot_value, " new index: ", idx - new_idx_unit)
        return query(query_number, query_list[:new_len], idx - new_idx_unit)
    else: 
        #print("query : ", query_number, " pivot: ", pivot_value, " new index: ", idx + new_idx_unit)
        return query(query_number, query_list[new_len:], idx + new_idx_unit)

