# -*- coding: utf-8 -*-
import numpy as np
import smuthi.particles as part
import smuthi.layers as lay
import smuthi.initial_field as init
import smuthi.coordinates as coord
import smuthi.simulation as simul
import smuthi.particle_coupling as coup
import smuthi.field_expansion as fldex

# Parameter input ----------------------------
vacuum_wavelength = 550
polar_angle = np.pi * 7/8
azimuthal_angle = np.pi * 1/3
polarization = 0
amplitude = 1
neff_waypoints = [0, 0.5, 0.8-0.01j, 2-0.01j, 2.5, 5]
neff_discr = 1e-2
lookup_resol = 5
# --------------------------------------------

coord.set_default_k_parallel(vacuum_wavelength, neff_waypoints, neff_discr)

# initialize particle object
sphere1 = part.Sphere(position=[0, 0, 100], refractive_index=2.4 + 0.0j, radius=100, l_max=3, m_max=3)
sphere2 = part.Sphere(position=[102, -100, 150], refractive_index=1.9 + 0.0j, radius=100, l_max=3, m_max=2)
sphere3 = part.Sphere(position=[202, 100, 250], refractive_index=1.7 + 0.0j, radius=100, l_max=2, m_max=2)
particle_list = [sphere1, sphere2, sphere3]

# initialize layer system object
lay_sys = lay.LayerSystem([0, 400, 0], [1.5, 2 + 0.01j, 1])

phi12 = np.arctan2(sphere1.position[1] - sphere2.position[1], sphere1.position[0] - sphere2.position[0])
rho12 = np.sqrt(sum([(sphere1.position[i]-sphere2.position[i])**2 for i in range(3)]))
sz12 = sphere1.position[2] + sphere2.position[2]
dz12 = sphere1.position[2] - sphere2.position[2]
wr12 = coup.layer_mediated_coupling_block(vacuum_wavelength, sphere1, sphere2, lay_sys)
w12 = coup.direct_coupling_block(vacuum_wavelength, sphere1, sphere2, lay_sys)

phi13 = np.arctan2(sphere1.position[1] - sphere3.position[1], sphere1.position[0] - sphere3.position[0])
rho13 = np.sqrt(sum([(sphere1.position[i]-sphere3.position[i])**2 for i in range(3)])) 
sz13 = sphere1.position[2] + sphere3.position[2]
dz13 = sphere1.position[2] - sphere3.position[2]
wr13 = coup.layer_mediated_coupling_block(vacuum_wavelength, sphere1, sphere3, lay_sys)
w13 = coup.direct_coupling_block(vacuum_wavelength, sphere1, sphere3, lay_sys)
 
# initialize initial field object
init_fld = init.PlaneWave(vacuum_wavelength=vacuum_wavelength, polar_angle=polar_angle,azimuthal_angle=azimuthal_angle, 
                          polarization=polarization, amplitude=amplitude)

# initialize simulation object
simulation_direct = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld,
                                    solver_type='LU', store_coupling_matrix=True)

simulation_lookup = simul.Simulation(layer_system=lay_sys, particle_list=particle_list, initial_field=init_fld,
                                     solver_type='gmres', store_coupling_matrix=False, 
                                     coupling_matrix_lookup_resolution=lookup_resol)
simulation_lookup.run()
coefficients_lookup = particle_list[0].scattered_field.coefficients
simulation_direct.run()
coefficients_direct = particle_list[0].scattered_field.coefficients

test_vec = np.ones(simulation_lookup.linear_system.master_matrix.shape)
M_lookup_test_vec = simulation_lookup.linear_system.coupling_matrix.linear_operator(test_vec)
M_direct_test_vec = simulation_direct.linear_system.coupling_matrix.linear_operator(test_vec)


def test_linear_operator():
    relerr = np.linalg.norm(M_lookup_test_vec - M_direct_test_vec) / np.linalg.norm(M_direct_test_vec)
    print('relative error linear operator: ', relerr)
    assert relerr < 5e-3
            
def test_result():
    relerr = np.linalg.norm(coefficients_lookup - coefficients_direct) / np.linalg.norm(coefficients_direct)
    print('relative error coefficient solution: ', relerr)
    assert relerr < 5e-4
    
if __name__ == '__main__':
    test_linear_operator()
    test_result()