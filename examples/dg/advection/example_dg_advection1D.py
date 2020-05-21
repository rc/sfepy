"""
Simple example for first order PDE

    du/dt - a * du/dx = 0

    u(t,0) = u(t,1)

    u(0, x) = ghump

"""
from examples.dg.example_dg_common import *

dim = 1

def define(filename_mesh=None, approx_order=2, flux=0, CFL=0.5, dt=None,
           Cw=None, diffusion_coef=None, diff_scheme_name="symmetric"):
    t0 = 0
    t1 = 1
    transient = True

    mstart = 0
    mend = 1

    diffusion_coef = None
    Cw = None

    example_name = "test_adv_1D"
    dim = 1

    if filename_mesh is None:
        filename_mesh = get_1Dmesh_hook(0, 1, 100)

    materials = {
        'a': ({'val': [1.0], '.flux': 0.0},),

    }

    regions = {
        'Omega': 'all',
        'Gamma': ('vertices of surface', 'facet'),
        'left': ('vertices in x == 0', 'vertex'),
        'right': ('vertices in x == 1', 'vertex')
    }

    fields = {
        'f': ('real', 'scalar', 'Omega', str(approx_order) + 'd', 'DG', 'legendre')
    }

    variables = {
        'u': ('unknown field', 'f', 0, 1),
        'v': ('test field', 'f', 'u'),
    }

    # dgebcs = {
    #     'u_left': ('left', {'u.all': 0}),
    #     'u_righ': ('right', {'u.all': 0}),
    # }

    dgepbc_1 = {
        'name'  : 'u_rl',
        'region': ['right', 'left'],
        'dofs': {'u.all': 'u.all'},
        'match': 'match_y_line',
    }

    integrals = {
        'i': 2 * approx_order,
    }

    equations = {
        'Advection': """
                       dw_volume_dot.i.Omega(v, u)
                       + dw_s_dot_mgrad_s.i.Omega(a.val, u[-1], v)
                       - dw_dg_advect_laxfrie_flux.i.Omega(a.val, v, u[-1]) = 0
                      """
    }

    solvers = {
        "tss": ('ts.tvd_runge_kutta_3',
                {"t0"     : t0,
                 "t1"     : t1,
                 # 'limiters': {"f": MomentLimiter1D},
                 'verbose': False}),
        'nls': ('nls.newton', {}),
        'ls' : ('ls.scipy_direct', {})
    }

    options = {
        'ts'              : 'tss',
        'nls'             : 'newton',
        'ls'              : 'ls',
        'save_times'      : 100,
        'active_only'     : False,
        'pre_process_hook': get_cfl_setup(CFL) if dt is None else get_cfl_setup(dt=dt),
        'output_format'   : "vtk"
    }

    functions = {}

    def local_register_function(fun):
        try:
            functions.update({fun.__name__: (fun,)})

        except AttributeError:  # Already a sfepy Function.
            fun = fun.function
            functions.update({fun.__name__: (fun,)})

        return fun

    @local_register_function
    def get_ic(x, ic=None):
        return four_step_u(x)

    def analytic_sol(coors, t=0):
        x = coors[..., 0]
        res = get_ic(x[..., None]) # - t[None, ...]) # for animating transient problem
        return res[..., 0]

    @local_register_function
    def sol_fun(ts, coors, mode="qp", **kwargs):
        t = ts.time
        if mode == "qp":
            return {"u": analytic_sol(coors, t)[..., None, None]}

    ics = {
        'ic': ('Omega', {'u.0': 'get_ic'}),
    }

    return locals()


globals().update(define())
