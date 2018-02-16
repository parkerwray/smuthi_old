# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:58:32 2018

@author: theobald2
"""

import numpy as np
import scipy
import smuthi.particles as part

def spheroid_orientation_generator(particle_number):
    
    euler_angles = np.zeros([particle_number, 3], dtype=float)
    for idx in range(particle_number):
        euler_angles[idx, 0] = np.random.rand() * 2 * np.pi
        euler_angles[idx, 1] = np.random.rand() * np.pi
        euler_angles[idx, 2] = float(0)
        
    return euler_angles

def collision_test(spheroid_nr1, spheroid_nr2):
    
    E1 = spheroid_nr1.spheroid_quadric_matrix()
    E2 = spheroid_nr2.spheroid_quadric_matrix()
    S = np.matrix.getH(np.linalg.cholesky(E1))
    
    ctr1, ctr2 = np.array(spheroid_nr1.position), np.array(spheroid_nr2.position)
    E2_prime = np.dot(np.transpose(np.linalg.inv(S)), np.dot(E2, np.linalg.inv(S)))
    ctr2_prime = -(np.dot(S, (ctr1 - ctr2)))  
    E2_prime_L = np.linalg.cholesky(E2_prime)
        
    H = np.dot(np.linalg.inv(E2_prime_L), np.transpose(np.linalg.inv(E2_prime_L)))
    p = np.array([0, 0, 0])
    f = np.dot(np.transpose(ctr2_prime - p), np.transpose(np.linalg.inv(E2_prime_L)))
    
    def minimization_fun(y_vec):
        fun = 0.5 * np.dot(np.dot(np.transpose(y_vec), H), y_vec) + np.dot(f, y_vec)
        return fun
    def constraint_fun(x):
        eq_constraint = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5 - 1
        return eq_constraint
    
    bnds = ((-1, 1), (-1, 1), (-1, 1))
    length_constraints = {'type' : 'eq', 'fun' : constraint_fun}
    
    flag = False
    while flag == False:
        x0 = -1 + np.dot((1 + 1), np.random.rand(3))
        optimization_result = scipy.optimize.minimize(minimization_fun, x0, method='SLSQP', bounds=bnds,
                                                      constraints=length_constraints, tol=None, callback=None, options=None)
        x_vec = np.transpose(np.dot(np.transpose(np.linalg.inv(E2_prime_L)), optimization_result['x'])
                             + np.transpose(ctr2_prime))
        if optimization_result['success'] == True:
            if np.linalg.norm(x_vec) <= 1:
                return True               
            elif np.linalg.norm(x_vec) < np.linalg.norm(ctr2_prime):
                flag = True
    return False

def circular_spheroid_sample_positions(particle_number, ensemble_radius, particle_list, z_limit, max_iter):
    
    cnt = 0
    particle_cnt = 0
    while cnt < max_iter and particle_cnt < particle_number:
        cnt += 1
        
        posx = (2 * np.random.rand() - 1) * ensemble_radius
        posy = (2 * np.random.rand() - 1) * ensemble_radius
        while posx ** 2 + posy ** 2 > ensemble_radius ** 2:
            posx = (2 * np.random.rand() - 1) * ensemble_radius
            posy = (2 * np.random.rand() - 1) * ensemble_radius
            
        posz = z_limit[particle_cnt, 0] + np.random.rand() * (z_limit[particle_cnt, 1] - z_limit[particle_cnt, 0]) 
        particle_list[particle_cnt].position = [posx, posy, posz]
        
        flag = False
        for idx in range(particle_cnt):
            if not flag :
                flag = collision_test(particle_list[particle_cnt], particle_list[idx]) 
        if not flag:
            particle_cnt += 1
            print('particle placed', particle_cnt)
        else: 
            print('particle not valid')

    if particle_cnt < particle_number:
        print('warning: not able to splace all spheroids')     

    return particle_list               
            
                
                


def spheroid_model_generator(particle_number, semi_axis_c, semi_axis_a, refractive_index, lmax, mmax, volume_density, scattering_layer, layer_system=None):
    
    ensemble_volume = 4/3 * np.pi * np.sum(np.dot(semi_axis_a ** 2,  semi_axis_c))
    particle_domain_volume = ensemble_volume / volume_density
    ensemble_radius = np.sqrt(particle_domain_volume / layer_system.thicknesses[scattering_layer])
    euler_angles_vec = spheroid_orientation_generator(particle_number)
    
    particle_list = []
    delta_z = np.zeros([particle_number, 2], dtype=float)
    z_limit = np.zeros([particle_number, 2], dtype=float)
    for idx in range(particle_number):
        particle_list.append(part.Spheroid(position=[0, 0, 0], euler_angles=euler_angles_vec[idx, :], refractive_index=refractive_index[idx],
                                           semi_axis_c=semi_axis_c[idx], semi_axis_a=semi_axis_a[idx], l_max=lmax[idx], m_max=mmax[idx]))
        r_high, r_low = particle_list[idx].spheroid_highest_lowest_surface_points() 
        delta_z[idx, 0], delta_z[idx, 1] = r_high[2], r_low[2]
    
    z_limit[:, 0] = np.sum(layer_system.thicknesses[:scattering_layer]) - delta_z[:, 1]
    z_limit[:, 1] = np.sum(layer_system.thicknesses[:(scattering_layer+1)]) - delta_z[:, 0]
    print(z_limit)
    return circular_spheroid_sample_positions(particle_number, ensemble_radius, particle_list, z_limit, max_iter=2000)




