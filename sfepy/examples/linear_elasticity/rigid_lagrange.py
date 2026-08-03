r"""
Linear elasticity example with rigid-body constraints enforced via Lagrange
multipliers.

Find :math:`\ul{u}`, :math:`\ul{\bar{u}}`, :math:`\ul{\bar{\omega}}`,
:math:`\ul{\lambda}` such that:

.. math::
    \int_{\Omega} D_{ijkl}\ e_{ij}(\ul{v}) e_{kl}(\ul{u})
    + \int_{\Gamma_R} \ul{v} \ul{\lambda}
    = 0
    \;, \quad \forall \ul{v} \;,

    - \int_{\Gamma_R} \ul{\bar{v}} \ul{\lambda}
    = 0
    \;, \quad \forall \ul{\bar{v}} \;,

    - \int_{\Gamma_R} (\ul{\bar{q}} \times (\ul{x} - \ul{x_c}) \ul{\lambda}
    = 0
    \;, \quad \forall \ul{\bar{q}} \;,

    + \int_{\Gamma_R} \ul{\nu} \ul{u}
    - \int_{\Gamma_R} \ul{\nu} \ul{\bar{u}}
    - \int_{\Gamma_R} \ul{\nu} (\ul{\bar{\omega}} \times (\ul{x} - \ul{x_c})
    = 0
    \;, \quad \forall \ul{\eta} \;.

The displacements in the ``'Rigid'`` region are constrained to rigid body
motions. The rigid volume, centered at :math:`\ul{x_c}`, is not present in
the domain's mesh, the constraint is applied to its surface :math:`\Gamma_R`.

Usage Examples
--------------

- Run with the default parameters, visualize deformation with 10 times
  magnified displacements::

    sfepy-run sfepy/examples/linear_elasticity/rigid_lagrange.py
    sfepy-view output/rigid_lagrange/cut-cylinder.*.vtk -f u:wu:f10:p0 1:vw:wu:f10:p0 1:vw:p0

- View the Lagrange multipliers :math:`\ul{\lambda}` and the rigid body motion
  displacements :math:`\ul{\bar{u}}` and rotations :math:`\ul{\bar{\omega}}`::

    sfepy-view output/rigid_lagrange/cut-cylinder.*.vtk -f lam:wu:f10:p0 1:vw:wu:f10:p0 1:vw:p0
    sfepy-view output/rigid_lagrange/cut-cylinder.*.vtk -f ub:wu:f10:p0 1:vw:wu:f10:p0 1:vw:p0
    sfepy-view output/rigid_lagrange/cut-cylinder.*.vtk -f ob:wu:f10:p0 1:vw:wu:f10:p0 1:vw:p0

- Change elasticity parameters and field approximation orders::

    sfepy-run sfepy/examples/linear_elasticity/rigid_lagrange.py -d "E=1e9,nu=0.3,order_u=2,order_lam=0"
"""
from functools import partial
import numpy as nm

from sfepy import data_dir
from sfepy.linalg import make_cross_matrices
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson

def get_pars(ts, coors, mode=None, rigid_centre=None, **kwargs):
    if mode == 'qp':
        mtxs = make_cross_matrices(coors - rigid_centre[None, :], first=False)
        return dict(C=mtxs)

def define(
        E=5e6,
        nu=0.45,
        move='x',
        shift=0.05,
        order_u=1,
        order_lam=1,
        refine=0,
        solver='auto',
        output_dir='output/rigid_lagrange',
        **kwargs,
):
    filename_mesh = data_dir + '/meshes/3d/cut-cylinder.vtk'

    rigid_centre = nm.array([0.0, 0.0, 0.0], dtype=nm.float64)

    cm, cs = -0.025, 0.025

    eps = 1e-8
    cm0, cm1 = cm - eps, cm + eps
    cs0, cs1 = cs - eps, cs + eps

    centre = [0, 0, 0]
    c = 'z'

    options = {
        'nls' : 'newton',
        'ls' : solver,
        'refinement_level' : refine,
        'output_dir': output_dir,
        'eterm': {
            'verbosity' : 0,
            'backend_args' : {
                'backend' : 'numpy',
                'optimize' : True,
                'layout' : None,
            },
        },
    }

    regions = {
        'Omega' :  'all',
        'Bottom' : (f'vertices in ({c} < -0.499999)', 'facet'),
        'Top' : (f'vertices in ({c} > 0.499999)', 'facet'),
        'RigidM' : (f'vertices in ({c} > {cm0}) & ({c} < {cm1})', 'facet'),
        'RigidS' : (f'vertices in ({c} > {cs0}) & ({c} < {cs1})', 'facet'),
        'Rigid' : ('r.RigidM +s r.RigidS', 'facet'),
    }

    fields = {
        'fu' : ('real', 'vector', 'Omega', order_u),
        'flag' : ('real', 'vector', 'Rigid', order_lam),
        'frig' : ('real', 'vector', 'Rigid', 0, 'L2', 'constant'),
    }

    variables = {
        'u' : ('unknown field', 'fu', 0),
        'v' : ('test field', 'fu', 'u'),
        'ub' : ('unknown field', 'frig', 1),
        'vb' : ('test field', 'frig', 'ub'),
        'ob' : ('unknown field', 'frig', 2),
        'qb' : ('test field', 'frig', 'ob'),
        'lam' : ('unknown field', 'flag', 3),
        'nu' : ('test field', 'flag', 'lam'),
    }

    materials = {
        'm' : ({
            'D' : stiffness_from_youngpoisson(3, young=E, poisson=nu),
        },),
        'r' : 'get_pars',
    }

    ebcs = {
        'bottom': ('Bottom', {'u.all': 0.0}),
        'top': ('Top', {'u.all': 'move_top'}),
    }

    def move_top(ts, coors, bc, problem, **kwargs):
        val = nm.zeros_like(coors)
        ic = dict(x=0, y=1, z=2)[move]
        val[:, ic] = ts.nt * shift
        return val

    functions = {
        'move_top' : (move_top,),
        'get_pars' : (partial(get_pars, rigid_centre=rigid_centre),),
    }

    integrals = {
        'i': 2 * max(order_u, order_lam),
    }

    equations = {
        'eq1' :
        """
        + dw_lin_elastic.i.Omega(m.D, v, u)
        + de_dot.i.Rigid(v, lam)
        = 0
        """,
        'eq2' :
        """
        - de_dot.i.Rigid(vb, lam)
        = 0
        """,
        'eq3' :
        """
        - de_dot.i.Rigid(r.C, qb, lam)
        = 0
        """,
        'eq4' :
        """
        + de_dot.i.Rigid(nu, u)
        - de_dot.i.Rigid(nu, ub)
        - de_dot.i.Rigid(r.C, nu, ob)
        = 0
        """,
    }

    solvers = {
        'pypardiso': ('ls.pypardiso', {}),
        'auto': ('ls.auto_direct', {
            'use_presolve' : True,
            'memory_relaxation' : 50,
        }),
        'newton' : ('nls.newton', {
            'i_max' : 1,
            'eps_a' : 1e-8,
            'lin_red' : None,
            'report_status' : True,
            'is_linear' : True,
        }),
        'ts' : ('ts.simple', {
            't0'     : 0.0,
            't1'     : 1.0,
            'dt'     : None,
            'n_step' : 5,
            'quasistatic' : True,
            'verbose' : 1,
        }),
    }

    return locals()
