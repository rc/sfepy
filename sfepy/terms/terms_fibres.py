import numpy as nm

from sfepy.base.base import Struct
from sfepy.terms.terms_hyperelastic_tl import HyperElasticTLBase
from sfepy.mechanics.tensors import dim2sym
from sfepy.homogenization.utils import iter_sym

def create_omega(fdir):
    r"""
    Create the fibre direction tensor :math:`\omega_{ij} = d_i d_j`.
    """
    n_el, n_qp, dim, _ = fdir.shape
    sym = dim2sym(dim)
    omega = nm.empty((n_el, n_qp, sym, 1), dtype=nm.float64)
    for ii, (ir, ic) in enumerate(iter_sym(dim)):
        omega[..., ii, 0] = fdir[..., ir, 0] * fdir[..., ic, 0]

    return omega

def compute_fibre_strain(green_strain, omega):
    """
    Compute the Green strain projected to the fibre direction.
    """
    eps = nm.zeros_like(omega[..., :1, :])
    for ii in range(omega.shape[2]):
        eps[..., 0, 0] += omega[..., ii, 0] * green_strain[..., ii, 0]

    return eps

def _setdefault_fibre_data(self, state):
    """
    Returns `fibre_data` :class:``Struct`` for storing/caching fibre-related
    data.
    """
    cache = self.set_default('fibre_cache', {})

    _, _, key = self.get_mapping(state, return_key=True)

    data_key = key + ('fibre_data',)
    if data_key in cache:
        fibre_data = cache[data_key]

    else:
        fibre_data = Struct()
        cache[data_key] = fibre_data

    return fibre_data

class FibresActiveTLTerm(HyperElasticTLBase):
    r"""
    Hyperelastic active fibres term. Effective stress
    :math:`S_{ij} = A f_{\rm max} \exp{\left\{-(\frac{\epsilon -
    \varepsilon_{\rm opt}}{s})^2\right\}} d_i d_j`,
    where :math:`\epsilon = E_{ij} d_i d_j` is the Green strain
    :math:`\ull{E}` projected to the fibre direction :math:`\ul{d}`.

    :Definition:

    .. math::
        \int_{\Omega} S_{ij}(\ul{u}) \delta E_{ij}(\ul{u};\ul{v})

    :Arguments:
        - material_1 : :math:`f_{\rm max}`
        - material_2 : :math:`\varepsilon_{\rm opt}`
        - material_3 : :math:`s`
        - material_4 : :math:`\ul{d}`
        - material_5 : :math:`A`
        - virtual    : :math:`\ul{v}`
        - state      : :math:`\ul{u}`
    """
    name = 'dw_tl_fib_a'
    arg_types = ('material_1', 'material_2', 'material_3',
                 'material_4', 'material_5', 'virtual', 'state')
    arg_shapes = {'material_1' : '1, 1', 'material_2' : '1, 1',
                  'material_3' : '1, 1', 'material_4' : 'D, 1',
                  'material_5' : '1, 1',
                  'virtual' : ('D', 'state'), 'state' : 'D'}
    family_data_names = ['green_strain']

    def get_fargs(self, mat1, mat2, mat3, mat4, mat5, virtual, state,
                  mode=None, term_mode=None, diff_var=None, **kwargs):
        fibre_data = _setdefault_fibre_data(self, state)

        fargs = HyperElasticTLBase.get_fargs(self,
                                             (mat1, mat2, mat3, mat4, mat5),
                                             virtual, state,
                                             mode, term_mode, diff_var,
                                             fibre_data=fibre_data,
                                             **kwargs)
        return fargs

    @staticmethod
    def stress_function(out, pars, green_strain,
                        fibre_data=None):
        fmax, eps_opt, s, fdir, act = pars

        omega = fibre_data.get('omega', None)
        if omega is None:
            omega = fibre_data.omega = create_omega(fdir)

        eps = fibre_data.eps = compute_fibre_strain(green_strain, omega)

        tau = fibre_data.tau = act * fmax * nm.exp(-((eps - eps_opt) / s)**2.0)

        out[:] = omega * tau

    @staticmethod
    def tan_mod_function(out, pars, green_strain,
                         fibre_data=None):
        fmax, eps_opt, s, fdir, act = pars

        omega, eps, tau = fibre_data.omega, fibre_data.eps, fibre_data.tau

        for ir in range(omega.shape[2]):
            for ic in range(omega.shape[2]):
                out[..., ir, ic] = omega[..., ir, 0] * omega[..., ic, 0]

        out[:] *= -2.0 * ((eps - eps_opt) / (s**2.0)) * tau

    def get_eval_shape(self, mat1, mat2, mat3, mat4, mat5, virtual, state,
                       mode=None, term_mode=None, diff_var=None, **kwargs):
        n_el, n_qp, dim, n_en, n_c = self.get_data_shape(state)
        sym = dim * (dim + 1) / 2

        return (n_el, 1, sym, 1), state.dtype

class FibresActive2TLTerm(HyperElasticTLBase):
    r"""
    Hyperelastic active fibres term. Effective stress
    :math:`S_{ij} = A f_{\rm max} \exp{\left\{-(\frac{\epsilon -
    \varepsilon_{\rm opt}}{s})^2\right\}} d_i d_j`,
    where :math:`\epsilon = E_{ij} d_i d_j` is the Green strain
    :math:`\ull{E}` projected to the fibre direction :math:`\ul{d}`.

    :Definition:

    .. math::
        \int_{\Omega} S_{ij}(\ul{u}) \delta E_{ij}(\ul{u};\ul{v})

    :Arguments:
        - material_1 : :math:`f_{\rm max}`
        - material_2 : :math:`\varepsilon_{\rm opt}`
        - material_3 : :math:`s`
        - material_4 : :math:`\ul{d}`
        - material_5 : :math:`A`
        - virtual    : :math:`\ul{v}`
        - state      : :math:`\ul{u}`
    """
    name = 'dw_tl_fib_a2'
    arg_types = ('ts', 'material_1', 'material_2', 'material_3',
                 'material_4', 'material_5', 'material_6',
                 'virtual', 'state')
    arg_shapes = {'material_1' : '1, 1', 'material_2' : '1, 1',
                  'material_3' : '1, 1', 'material_4' : 'D, 1',
                  'material_5' : '1, 1',
                  'virtual' : ('D', 'state'), 'state' : 'D'}
    family_data_names = ['green_strain']

    def get_fargs(self, ts, mat1, mat2, mat3, mat4, mat5, mat6, virtual, state,
                  mode=None, term_mode=None, diff_var=None, **kwargs):
        fibre_data = _setdefault_fibre_data(self, state)

        fargs = HyperElasticTLBase.get_fargs(self,
                                             (mat1, mat2, mat3, mat4,
                                              mat5, mat6),
                                             virtual, state,
                                             mode, term_mode, diff_var,
                                             ts=ts,
                                             fibre_data=fibre_data,
                                             **kwargs)

        return fargs

    @staticmethod
    def stress_function(out, pars, green_strain,
                        ts=None, fibre_data=None):
        fmax, eps_opt, s, fdir, act, eta = pars

        omega = fibre_data.get('omega', None)
        if omega is None:
            omega = fibre_data.omega = create_omega(fdir)

        print ts
        print fibre_data

        from sfepy.base.base import debug; debug()

        # Compute fibre strain(s) and stress, store them in fibre_data
        # eps = ...
        # deps = ... depending if ts.step > 0 or not.
        # tau = ...

        out[:] = omega * tau

    @staticmethod
    def tan_mod_function(out, pars, green_strain,
                         ts=None, fibre_data=None):
        fmax, eps_opt, s, fdir, act, eta = pars

        # Reuse data...
        # omega, eps, tau, ... = fibre_data.omega, fibre_data.eps, fibre_data.tau

        for ir in range(omega.shape[2]):
            for ic in range(omega.shape[2]):
                out[..., ir, ic] = omega[..., ir, 0] * omega[..., ic, 0]

        # Compute tau derivative...
        #  out[:] *= ...

    def get_eval_shape(self, ts, mat1, mat2, mat3, mat4, mat5, mat6,
                       virtual, state,
                       mode=None, term_mode=None, diff_var=None, **kwargs):
        n_el, n_qp, dim, n_en, n_c = self.get_data_shape(state)
        sym = dim * (dim + 1) / 2

        return (n_el, 1, sym, 1), state.dtype
