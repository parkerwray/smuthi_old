# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:38:51 2018

@author: theobald2
"""
import numpy as np
import smuthi.coordinates as coord
import smuthi.field_expansion as fldex
    
    
def fieldpoints(xmin, xmax, ymin, ymax, zmin, zmax, resolution):
    """
    Creates an 4-dimensional array to handle the nearfield computation via plane waves
    Args:
        xmin (float):       Plot from that x (length unit)
        xmax (float):       Plot up to that x (length unit)
        ymin (float):       Plot from that y (length unit)
        ymax (float):       Plot up to that y (length unit)
        zmin (float):       Plot from that z (length unit)
        zmax (float):       Plot up to that z (length unit)
        resolution (float):     Compute the field with that spatial resolution (length unit)
    Returns:
        fp (numpy.array):       4-dimensional array 
                                dimension 1,2,3 contains the x,y,z coordinate of all field points
                                dimension 4 contains a 'bool' whether E has been computed or not
    """      
    if xmin == xmax:
        dim1vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        yarr, zarr = np.meshgrid(dim1vec, dim2vec)
        xarr = yarr - yarr + xmin
    elif ymin == ymax:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(zmin, zmax, (zmax - zmin) / resolution + 1, endpoint=True)
        xarr, zarr = np.meshgrid(dim1vec, dim2vec)
        yarr = xarr - xarr + ymin
    else:
        dim1vec = np.linspace(xmin, xmax, (xmax - xmin) / resolution + 1, endpoint=True)
        dim2vec = np.linspace(ymin, ymax, (ymax - ymin) / resolution + 1, endpoint=True)
        xarr, yarr = np.meshgrid(dim1vec, dim2vec)
        zarr = xarr - xarr + zmin

    fp = np.empty((dim1vec.size, dim2vec.size, 4), dtype=object)   
    fp[:, :, 0], fp[:, :, 1], fp[:, :, 2] = xarr, yarr, zarr
    fp[:, :, 3] = np.zeros((dim1vec.size, dim2vec.size), dtype=bool)
    
    return fp, dim1vec, dim2vec 
    
def inside_particles(fieldpoints, particle_list):
    """
    Checks for each field point, whether it is located inside one of the particles.
    Args:
        fieldpoints(numpy.array):       4-dimensional numpy.array that contains the x,y,z-coordinate of each fieldpoint 
                                        and a bool, to determin wether this fieldpoint needs to be computed
        particle_list:                  Object of smuthi.particle_list that contains all information about the scattering particles.
    Returns:
        fieldpoints(numpy.array):       The 4th dimension now contains the information whether the coordinate is inside a particle.
                                        If so, the scattered field does not need to be computed.
    """
        
    for p in range(np.size(particle_list)):    
        E = particle_list[p].spheroid_quadric_matrix()
        L = np.linalg.cholesky(E)
        S = np.matrix.getH(L)
        for k in range(np.size(fieldpoints[:, 0, 0])):
            for l in range(np.size(fieldpoints[0, :, 0])):
                if not fieldpoints[k, l, 3]:
                    coord_prime = -(np.dot(S, (particle_list[p].position - np.array([fieldpoints[k, l, 0], fieldpoints[k, l, 1], fieldpoints[k, l, 2]]))))
                    if np.round(np.linalg.norm(coord_prime), 5) <= 1:
                        fieldpoints[k, l, 3] = 'True'

    return fieldpoints    

# I am not sure, what happens when a particle is located near an interface. In this case a fieldpoint might be inside the circumscribing sphere and beyond
# the interface. 
# Additonally points that are inside the circumscribing sphere and in the same layer might be beyond an interface after rotation of the coordinate system.
def pwe_nearfield_superposition(xmin, xmax, ymin, ymax, zmin, zmax, resolution, k_parallel='default', azimuthal_angles='default',
                       simulation=None):
    """
    Computes the electric field at an 2-dimensionaly array of locations. 
    Outside the circumscribing sphere of a spheroid, the SWE is used. Inside, PWEs are utilized.
    Args:
        xmin, ymin, zmin (float):           minimal x/y/z-value of the the 2-dimensional array
        xmax, ymax, zmax (float):           maximal x/y/z-value of the the 2-dimensional array / one has to be equal to its minimal value
        resolution (int):                   distance between data points
        k_parallel (numpy.ndarray or str):  in-plane wavenumber. If 'default', use smuthi.coord.default_k_parallel
        azimuthal_angles (numpy.ndarray or str):    azimuthal angle values (radian) if 'default', use smuthi.coordinates.default_azimuthal_angles
    Returns:
        right now too much
    """    
    fp0, dim1vec, dim2vec = fieldpoints(xmin, xmax, ymin, ymax, zmin, zmax, resolution)
    field_list = np.empty((np.size(simulation.particle_list), dim1vec.size, dim2vec.size, 4), dtype=object)
    Ex = np.zeros([np.size(dim1vec), np.size(dim2vec)], dtype=complex)
    Ey, Ez = Ex, Ex
    fp = inside_particles(fp0, simulation.particle_list)
    for i in range(np.size(simulation.particle_list)):
        print('Nearfield for spheroid', i)
        field_list[i, :, :, 0] = np.zeros((dim1vec.size, dim2vec.size), dtype=complex)
        field_list[i, :, :, 1], field_list[i, :, :, 2] = field_list[i, :, :, 0], field_list[i, :, :, 0]        
        field_list[i, :, :, 3] = fp[:, :, 3]
# all points outside the circumscribing sphere
        temp_array = np.array([0, 0, 0, 0, 0], dtype=float)[None,:]
        for k in range(np.size(fp[:, 0, 0])):
            for l in range(np.size(fp[0, :, 0])):
                if not field_list[i, k, l, 3]:
                    if (np.linalg.norm(np.abs(np.array([fp[k, l, 0] - simulation.particle_list[i].position[0], 
                                                        fp[k, l, 1] - simulation.particle_list[i].position[1],
                                                        fp[k, l, 2] - simulation.particle_list[i].position[2]])))
                        / np.max([simulation.particle_list[i].semi_axis_a, simulation.particle_list[i].semi_axis_c])) >= 1: # this should be a 1
                       temp_array = np.append(temp_array, np.array([k , l, fp[k, l, 0], fp[k, l, 1], fp[k, l, 2]])[None,:],
                                              axis=0)

        temp_array = np.delete(temp_array, 0, 0)
        ex, ey, ez = (simulation.particle_list[i].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0],
                      simulation.particle_list[i].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1], 
                      simulation.particle_list[i].scattered_field.electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2])
        for p in range(np.size(temp_array, axis=0)):
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 0] = ex[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 1] = ey[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 2] = ez[p]
            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 3] = 'True'
        if ex.any():
            del ex, ey, ez               
# all points inside the circumscribing sphere and above/below the highest/lowest point of the spheroid
#        r_highz, r_lowz = simulation.particle_list[i].spheroid_highest_lowest_surface_points()
#        temp_array = np.array([0, 0, 0, 0, 0], dtype=float)[None,:]
#        for k in range(np.size(fp[:, 0, 0])):
#            for l in range(np.size(fp[0, :, 0])):
#                if not field_list[i, k, l, 3]:
#                    if fp[k, l, 2] <= r_lowz[2] or fp[k, l, 2] >= r_highz[2]:
#                        temp_array = np.append(temp_array, np.array([k, l, fp[k, l, 0], fp[k, l, 1], fp[k, l, 2]])[None,:], axis=0)
#        
#        temp_array = np.delete(temp_array, 0, 0)
#        pwe = fldex.swe_to_pwe_conversion(swe=simulation.particle_list[i].scattered_field, k_parallel='default', azimuthal_angles='default',
#                                          layer_system=simulation.layer_system,
#                                          layer_number=simulation.layer_system.layer_number(simulation.particle_list[i].position[2]),
#                                          layer_system_mediated=False)
#
#        ex, ey, ez = (pwe[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0]
#                      + pwe[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0],
#                      pwe[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1]
#                      + pwe[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1], 
#                      pwe[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2]
#                      + pwe[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2])      
#        for p in range(np.size(temp_array, axis=0)):
#            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 0] = ex[p]
#            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 1] = ey[p]
#            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 2] = ez[p]
#            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 3] = 'True'
#        if ex.any():
#            del ex, ey, ez   
        
# all points that are inside the circumscribing sphere and that need a rotation        
        while not np.array(field_list[i, :, :, 3], dtype=bool).all():
            for k in range(np.size(fp[:, 0, 0])):
                for l in range(np.size(fp[0, :, 0])):
                    if not field_list[i, k, l, 3]:
                        coordinate = np.array([fp[k, l, 0], fp[k, l, 1], fp[k, l, 2]])
                        p1, alpha, beta = simulation.particle_list[i].spheroid_closest_surface_point(coordinate)
                        r_highz, r_lowz = simulation.particle_list[i].spheroid_highest_lowest_surface_points(add_rot=[-alpha, -beta, 0])
                        b_prime = np.dot(np.transpose(fldex.block_rotation_matrix_D_svwf(simulation.particle_list[i].l_max,
                                         simulation.particle_list[i].m_max, -alpha, -beta, 0)), simulation.particle_list[i].scattered_field.coefficients)
                        swe_prime = fldex.SphericalWaveExpansion(k=simulation.particle_list[i].scattered_field.k,
                                                                 l_max=simulation.particle_list[i].scattered_field.l_max,
                                                                 m_max=simulation.particle_list[i].scattered_field.m_max, kind='outgoing', 
                                                                 reference_point=simulation.particle_list[i].position, lower_z=-np.inf, upper_z=np.inf, 
                                                                 inner_r=0, outer_r=np.inf)
                        swe_prime.coefficients = b_prime
                        pwe_prime = fldex.swe_to_pwe_conversion(swe=swe_prime, k_parallel='default', azimuthal_angles='default',
                                                                layer_system=simulation.layer_system,
                                                                layer_number=simulation.layer_system.layer_number(simulation.particle_list[i].position[2]),
                                                                layer_system_mediated=False)
                        
                        temp_array = np.array([0, 0, 0, 0, 0], dtype=float)[None,:]
                        for m in range(np.size(fp[:, 0, 0])):
                            for n in range(np.size(fp[0, :, 0])):
                                if not field_list[i, m, n, 3]:
                                    coordref = (np.array([fp[m, n, 0], fp[m, n, 1], fp[m, n, 2]]) - np.array(simulation.particle_list[i].position))
                                    coordref_prime = coord.vector_rotation(coordref, euler_angles=[-alpha, -beta, 0])
                                    coord_prime = coordref_prime + simulation.particle_list[i].position
                                    if coord_prime[2] <= r_lowz[2] or coord_prime[2] >= r_highz[2]:
                                        temp_array = np.append(temp_array, np.array([m, n, coord_prime[0], coord_prime[1], coord_prime[2]])[None,:],
                                                               axis=0)
                        temp_array = np.delete(temp_array, 0, 0)                        
                        ex, ey, ez = (pwe_prime[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0]
                                      + pwe_prime[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[0],
                                      pwe_prime[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1]
                                      + pwe_prime[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[1], 
                                      pwe_prime[0].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2]
                                      + pwe_prime[1].electric_field(temp_array[:, 2], temp_array[:, 3], temp_array[:, 4])[2])
                        for p in range(np.size(temp_array, axis=0)):
                            E_prime = coord.inverse_vector_rotation(np.array([ex[p], ey[p], ez[p]]), euler_angles=[-alpha, -beta, 0])
                            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 0] = E_prime[0]    
                            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 1] = E_prime[1]
                            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 2] = E_prime[2]
                            field_list[i, int(temp_array[p, 0]), int(temp_array[p, 1]), 3] = 'True'
                        if ex.any():
                            del ex, ey, ez, E_prime
                   
        Ex = Ex + np.array(field_list[i, :, :, 0], dtype=complex)
        Ey = Ey + np.array(field_list[i, :, :, 1], dtype=complex)
        Ez = Ez + np.array(field_list[i, :, :, 2], dtype=complex)
    
    
    
    return field_list, Ex, Ey, Ez, fp0, dim1vec, dim2vec

