# -*- coding: utf-8 -*-
"""Provide class for the representation of scattering particles."""
import numpy as np
import collections
import scipy


class Particle:
    """Base class for scattering particles.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        euler_angles (list):        Particle Euler angles in the format [alpha, beta, gamma]
        refractive_index (complex): Complex refractive index of particle
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
    """
    def __init__(self, position=None, euler_angles=None, refractive_index=1+0j, l_max=None, m_max=None):
        
        if position is None:
            self.position = [0, 0, 0]
        else:
            self.position = position

        if euler_angles is None:
            self.euler_angles = [0, 0, 0]
        else:
            self.euler_angles = euler_angles

        self.refractive_index = refractive_index
        self.l_max = l_max
        if m_max is not None:
            self.m_max = m_max
        else:
            self.m_max = l_max
        self.initial_field = None
        self.scattered_field = None
        self.t_matrix = None
        
    def circumscribing_sphere_radius(self):
        """Virtual method to be overwritten"""
        pass

    def circumscribing_sphere_intersection(self, particle_prime):
        """Virtual method to be overwritten"""
        pass    

class Sphere(Particle):
    """Particle subclass for spheres.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        refractive_index (complex): Complex refractive index of particle
        radius (float):             Particle radius (length unit)
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
        t_matrix_method (dict):     Dictionary containing the parameters for the algorithm to compute the T-matrix
    """
    def __init__(self, position=None, refractive_index=1+0j, radius=1, l_max=None, m_max=None):

        Particle.__init__(self, position=position, refractive_index=refractive_index, l_max=l_max, m_max=m_max)

        self.radius = radius
        
    def circumscribing_sphere_radius(self):
        return self.radius

    def circumscribing_sphere_intersection(self, particle_prime):
        """Check, whether the particle intersects another particle's circumscribing sphere."""
        distance = np.linalg.norm(np.array(self.position) - np.array(particle_prime.position))
        if distance > self.circumscribing_sphere_radius() + particle_prime.circumscribing_sphere_radius():
            return False
        else:
            if type(particle_prime).__name__ == 'Sphere':
                raise ValueError('Spheres' + self + 'and' + particle_prime + 'intersect')
            else:
                return True

class Spheroid(Particle):
    """Particle subclass for spheroids.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        euler_angles (list):        Euler angles [alpha, beta, gamma] in (zy'z''-convention) in radian.
                                    Alternatively, you can specify the polar and azimuthal angle of the axis of 
                                    revolution.
        polar_angle (float):        Polar angle of axis of revolution. 
        azimuthal_angle (float):    Azimuthal angle of axis of revolution.
        refractive_index (complex): Complex refractive index of particle
        semi_axis_c (float):        Spheroid half axis in direction of axis of revolution (z-axis if not rotated)
        semi_axis_a (float):        Spheroid half axis in lateral direction (x- and y-axis if not rotated)
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
        t_matrix_method (dict):     Dictionary containing the parameters for the algorithm to compute the T-matrix
    """
    def __init__(self, position=None, euler_angles=None, polar_angle=0, azimuthal_angle=0, refractive_index=1+0j, 
                 semi_axis_c=1, semi_axis_a=1, l_max=None, m_max=None, t_matrix_method=None):

        if euler_angles is None:
            euler_angles = [azimuthal_angle, polar_angle, 0]
            
        Particle.__init__(self, position=position, euler_angles=euler_angles, refractive_index=refractive_index,
                          l_max=l_max, m_max=m_max)
        
        if t_matrix_method is None:
            self.t_matrix_method = {}
        else:
            self.t_matrix_method = t_matrix_method

        self.semi_axis_c = semi_axis_c
        self.semi_axis_a = semi_axis_a

    def circumscribing_sphere_radius(self):
        return max([self.semi_axis_a, self.semi_axis_c])

    def circumscribing_sphere_intersection(self, particle_prime):
        """Check, whether the particle intersects another particle's circumscribing sphere."""
        distance = np.linalg.norm(np.array(self.position) - np.array(particle_prime.position))
        if distance > self.circumscribing_sphere_radius() + particle_prime.circumscribing_sphere_radius():
            return False
        else:
            if type(particle_prime).__name__ == 'Sphere':
                return True
            elif type(particle_prime).__name__ == 'Spheroid':
                closest_point = self.spheroid_closest_surface_point(np.array(particle_prime.position))
                if np.linalg.norm(closest_point - particle_prime.position) > particle_prime.circumscribing_sphere_radius():
                    return False
                else:
                    return True
            elif type(particle_prime).__name__ == 'FiniteCylinder':
                raise ValueError('For finite cylinders the circumscribing sphere routine is not implemented yet.')
                
    
    def spheroid_quadric_matrix(self):
        """
        Computation of the (3x3)-matrix E, that represents a spheroid.
        The eigenvalues of this matrix are determined by the spheroid's semi-axis.
        The eigenvectors of this matrix are determined by the spheroid's orientation.
        Returns:
            E(numpy.array):      (3x3)-matrix. Complete description of a spheroid.
        """
        def rotation_matrix(ang):
            rot_mat = (np.array([[np.cos(ang[0]) * np.cos(ang[1]), -np.sin(ang[0]), np.cos(ang[0]) * np.sin(ang[1])],
                                 [np.sin(ang[0]) * np.cos(ang[1]), np.cos(ang[0]), np.sin(ang[0]) * np.sin(ang[1])],
                                 [-np.sin(ang[1]), 0, np.cos(ang[1])]]))
            return rot_mat
    
        rot_matrix = rotation_matrix(self.euler_angles)
        eigenvalue_matrix = np.array([[1 / self.semi_axis_a ** 2, 0, 0], [0, 1 / self.semi_axis_a ** 2, 0], [0, 0, 1 / self.semi_axis_c ** 2]])
        E = np.dot(rot_matrix, np.dot(eigenvalue_matrix, np.transpose(rot_matrix)))
        return E
    
       
    def spheroid_closest_surface_point(self, coordinate):
        """
        Computation of a spheroids surface point, that is closest to a given reference coordinate   
        Args:
            An smuthi.spheroid-Object
            coordinate (numpy.array):    Reference point        
        Retruns:
                - surface point closest to the reference coordinate (numpy.array)
        """     
        position = np.array(self.position)
        E = self.spheroid_quadric_matrix()
        L = np.linalg.cholesky(E)   
        S = np.matrix.getH(L)
        if np.round(np.linalg.norm(-(np.dot(S, (position - coordinate)))), 5) <= 1:
            raise ValueError('The given point is located inside the spheroid')
    
        H = np.dot(np.linalg.inv(L), np.transpose(np.linalg.inv(L)))
        f = np.dot(np.transpose(position - coordinate), np.transpose(np.linalg.inv(L)))
            
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
            p1 = np.transpose(np.dot(np.transpose(np.linalg.inv(L)), optimization_result['x']) + np.transpose(position))
            if optimization_result['success'] == True:
                if np.linalg.norm(p1 - coordinate) < np.linalg.norm(position - coordinate):
                    flag = True
                else:
                    print('wrong minimum ...')
            else:
                print('No minimum found ...')
        return p1
    
    

class FiniteCylinder(Particle):
    """Particle subclass for finite cylinders.

    Args:
        position (list):            Particle position in the format [x, y, z] (length unit)
        euler_angles (list):        Euler angles [alpha, beta, gamma] in (zy'z''-convention) in radian.
                                    Alternatively, you can specify the polar and azimuthal angle of the axis of 
                                    revolution.
        polar_angle (float):        Polar angle of axis of revolution. 
        azimuthal_angle (float):    Azimuthal angle of axis of revolution.
        refractive_index (complex): Complex refractive index of particle
        cylinder_radius (float):    Radius of cylinder (length unit)
        cylinder_height (float):    Height of cylinder, in z-direction if not rotated (length unit)
        l_max (int):                Maximal multipole degree used for the spherical wave expansion of incoming and
                                    scattered field
        m_max (int):                Maximal multipole order used for the spherical wave expansion of incoming and
                                    scattered field
    """
    def __init__(self, position=None, euler_angles=None, polar_angle=0, azimuthal_angle=0, refractive_index=1+0j, 
                 cylinder_radius=1, cylinder_height=1, l_max=None, m_max=None, t_matrix_method=None):

        if euler_angles is None:
            euler_angles = [azimuthal_angle, polar_angle, 0]

        Particle.__init__(self, position=position, euler_angles=euler_angles, refractive_index=refractive_index,
                          l_max=l_max, m_max=m_max)

        if t_matrix_method is None:
            self.t_matrix_method = {}
        else:
            self.t_matrix_method = t_matrix_method

        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        
    def circumscribing_sphere_radius(self):
        return np.sqrt((self.cylinder_height / 2)**2 + self.cylinder_radius**2)

    def circumscribing_sphere_intersection(self, particle_prime):
        """Check, whether the particle intersects another particle's circumscribing sphere."""
        distance = np.linalg.norm(np.array(self.position) - np.array(particle_prime.position))
        if distance > self.circumscribing_sphere_radius() + particle_prime.circumscribing_sphere_radius():
            return False
        else:
            if type(particle_prime).__name__ == 'Sphere':
                return True
            else:
                raise ValueError('For finite cylinders the circumscribing sphere routine is not implemented yet.')

class ParticleList(collections.UserList):
    """A class to handle particle collections. Besides holding a list of particles object, methods to check particle
     intersection are provided.
     """
    def __init__(self, lst=None):
        if lst is None:
            lst = []
        collections.UserList.__init__(self, lst)