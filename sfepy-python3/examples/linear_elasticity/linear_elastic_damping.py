r"""
Time-dependent linear elasticity with a simple damping.

Find :math:`\ul{u}` such that:

.. math::
    \int_{\Omega} c\ \ul{v} \cdot \pdiff{\ul{u}}{t}
    + \int_{\Omega} D_{ijkl}\ e_{ij}(\ul{v}) e_{kl}(\ul{u})
    = 0
    \;, \quad \forall \ul{v} \;,

where

.. math::
    D_{ijkl} = \mu (\delta_{ik} \delta_{jl}+\delta_{il} \delta_{jk}) +
    \lambda \ \delta_{ij} \delta_{kl}
    \;.
"""
import numpy as nm
from .linear_elastic import \
     filename_mesh, materials, regions, fields, ebcs, \
     integrals, solvers

def print_times(problem, state):
    print(nm.array(problem.ts.times))

options = {
    'ts' : 'ts',
    'save_steps' : -1,
    'post_process_hook_final' : print_times,
}

variables = {
    'u' : ('unknown field', 'displacement', 0, 1),
    'v' : ('test field', 'displacement', 'u'),
}

# Put density to 'solid'.
materials['solid'][0].update({'c' : 1000.0})

# Moving the PerturbedSurface region.
ebcs['PerturbedSurface'][1].update({'u.0' : 'ebc_sin'})

def ebc_sin(ts, coors, **kwargs):
    val = 0.01 * nm.sin(2.0*nm.pi*ts.nt)
    return nm.tile(val, (coors.shape[0],))

equations = {
    'balance_of_forces in time' :
    """dw_volume_dot.i1.Omega( solid.c, v, du/dt )
     + dw_lin_elastic_iso.i1.Omega( solid.lam, solid.mu, v, u ) = 0""",
}

solvers.update({
    'ts' : ('ts.adaptive', {
        't0' : 0.0,
        't1' : 1.0,
        'dt' : None,
        'n_step' : 101,

        'adapt_fun' : 'adapt_time_step',
    }),
})

def adapt_time_step(ts, status, adt, problem):
    if ts.time > 0.5:
        ts.set_time_step(0.1)

    return True

# Pre-assemble and factorize the matrix prior to time-stepping.
newton = solvers['newton']
newton[1].update({'problem' : 'nonlinear'}) # Change of time step changes
                                            # matrix!

ls = solvers['ls']
ls[1].update({'presolve' : True})

functions = {
    'ebc_sin' : (ebc_sin,),
    'adapt_time_step' : (adapt_time_step,),
}
