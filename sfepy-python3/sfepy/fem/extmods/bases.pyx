# -*- Mode: Python -*-
"""
Polynomial base functions and related utilities.
"""
cimport cython

cimport numpy as np
import numpy as np

cimport _fmfield as _f
from _fmfield cimport FMField

from types cimport int32, float64, complex128

cdef extern from 'math.h':
    cdef float64 sqrt(float x)

cdef extern from 'common.h':
    void *pyalloc(size_t size)
    void pyfree(void *pp)
    int Max_i 'Max'(int a, int b)
    double Max_f 'Max'(double a, double b)
    double Min_f 'Min'(double a, double b)

cdef extern from 'lagrange.h':
    int32 _get_barycentric_coors \
          'get_barycentric_coors'(FMField *bc, FMField *coors, FMField *mtx_i,
                                  float64 eps, int32 check_errors)

    int32 _get_xi_simplex \
          'get_xi_simplex'(FMField *xi, FMField *bc, FMField *dest_point,
                           FMField *ref_coors, FMField *e_coors)

    int32 _get_xi_tensor \
          'get_xi_tensor'(FMField *xi,
                          FMField *dest_point, FMField *e_coors,
                          FMField *mtx_i,
                          FMField *base1d, int32 *nodes, int32 n_col,
                          float64 vmin, float64 vmax,
                          int32 i_max, float64 newton_eps)

    int32 _eval_lagrange_simplex \
          'eval_lagrange_simplex'(FMField *out, FMField *bc, FMField *mtx_i,
                                  int32 *nodes, int32 n_col,
                                  int32 order, int32 diff)

    int32 _eval_lagrange_tensor_product \
          'eval_lagrange_tensor_product'(FMField *out, FMField *bc,
                                         FMField *mtx_i, FMField *base1d,
                                         int32 *nodes, int32 n_col,
                                         int32 order, int32 diff)

@cython.boundscheck(False)
def get_barycentric_coors(np.ndarray[float64, mode='c', ndim=2] coors not None,
                          np.ndarray[float64, mode='c', ndim=2] mtx_i not None,
                          float64 eps=1e-8,
                          int check_errors=False):
    """
    Get barycentric (area in 2D, volume in 3D) coordinates of points.

    Parameters
    ----------
    coors : array
        The coordinates of the points, shape `(n_coor, dim)`.
    mtx_i : array
        The inverse of simplex coordinates matrix, shape `(dim + 1, dim + 1)`.
    eps : float
        The tolerance for snapping out-of-simplex point back to the simplex.
    check_errors : bool
        If True, raise ValueError if a barycentric coordinate is outside
        the snap interval `[-eps, 1 + eps]`.

    Returns
    -------
    bc : array
        The barycentric coordinates, shape `(n_coor, dim + 1)`. Then
        reference element coordinates `xi = dot(bc, ref_coors)`.
    """
    cdef int n_coor = coors.shape[0]
    cdef int n_v = mtx_i.shape[0]
    cdef FMField _bc[1], _coors[1], _mtx_i[1]
    cdef np.ndarray[float64, ndim=2] bc = np.zeros((n_coor, n_v),
                                                   dtype=np.float64)

    _f.array2fmfield2(_bc, bc)
    _f.array2fmfield2(_coors, coors)
    _f.array2fmfield2(_mtx_i, mtx_i)
    _get_barycentric_coors(_bc, _coors, _mtx_i, eps, check_errors)
    return bc

@cython.boundscheck(False)
def eval_lagrange_simplex(np.ndarray[float64, mode='c', ndim=2] coors not None,
                          np.ndarray[float64, mode='c', ndim=2] mtx_i not None,
                          np.ndarray[int32, mode='c', ndim=2] nodes not None,
                          int order, int diff=False,
                          float64 eps=1e-15,
                          int check_errors=True):
    """
    Evaluate Lagrange base polynomials in given points on simplex domain.

    Parameters
    ----------
    coors : array
        The coordinates of the points, shape `(n_coor, dim)`.
    mtx_i : array
        The inverse of simplex coordinates matrix, shape `(dim + 1, dim + 1)`.
    nodes : array
        The description of finite element nodes, shape `(n_nod, dim + 1)`.
    order : int
        The polynomial order.
    diff : bool
        If True, return base function derivatives.
    eps : float
        The tolerance for snapping out-of-simplex point back to the simplex.
    check_errors : bool
        If True, raise ValueError if a barycentric coordinate is outside
        the snap interval `[-eps, 1 + eps]`.

    Returns
    -------
    out : array
        The evaluated base functions, shape `(n_coor, 1 or dim, n_nod)`.
    """
    cdef int bdim
    cdef int n_coor = coors.shape[0]
    cdef int dim = mtx_i.shape[0] - 1
    cdef int n_nod = nodes.shape[0]
    cdef FMField _out[1], _bc[1], _coors[1], _mtx_i[1]
    cdef int32 *_nodes = &nodes[0, 0]
    cdef np.ndarray[float64, ndim=2] bc = np.zeros((n_coor, dim + 1),
                                                   dtype=np.float64)

    assert mtx_i.shape[0] == nodes.shape[1]

    if diff:
        bdim = dim

    else:
        bdim = 1

    cdef np.ndarray[float64, ndim=3] out = np.zeros((n_coor, bdim, n_nod),
                                                    dtype=np.float64)

    _f.array2fmfield3(_out, out)
    _f.array2fmfield2(_bc, bc)
    _f.array2fmfield2(_coors, coors)
    _f.array2fmfield2(_mtx_i, mtx_i)

    _get_barycentric_coors(_bc, _coors, _mtx_i, eps, check_errors)
    _eval_lagrange_simplex(_out, _bc, _mtx_i, _nodes, nodes.shape[1],
                           order, diff)

    return out

@cython.boundscheck(False)
def eval_lagrange_tensor_product(np.ndarray[float64, mode='c', ndim=2]
                                 coors not None,
                                 np.ndarray[float64, mode='c', ndim=2]
                                 mtx_i not None,
                                 np.ndarray[int32, mode='c', ndim=2]
                                 nodes not None,
                                 int order, int diff=False,
                                 float64 eps=1e-15,
                                 int check_errors=True):
    """
    Evaluate Lagrange base polynomials in given points on tensor product
    domain.

    Parameters
    ----------
    coors : array
        The coordinates of the points, shape `(n_coor, dim)`.
    mtx_i : array
        The inverse of 1D simplex coordinates matrix, shape `(2, 2)`.
    nodes : array
        The description of finite element nodes, shape `(n_nod, 2 * dim)`.
    order : int
        The polynomial order.
    diff : bool
        If True, return base function derivatives.
    eps : float
        The tolerance for snapping out-of-simplex point back to the simplex.
    check_errors : bool
        If True, raise ValueError if a barycentric coordinate is outside
        the snap interval `[-eps, 1 + eps]`.

    Returns
    -------
    out : array
        The evaluated base functions, shape `(n_coor, 1 or dim, n_nod)`.
    """
    cdef int ii, idim, im, ic
    cdef int n_coor = coors.shape[0]
    cdef int n_nod = nodes.shape[0]
    cdef int dim = coors.shape[1]
    cdef FMField _out[1], _bc[1], _coors[1], _mtx_i[1], _base1d[1]
    cdef int32 *_nodes = &nodes[0, 0]
    cdef np.ndarray[float64, ndim=3] bc = np.zeros((dim, n_coor, 2),
                                                   dtype=np.float64)
    cdef np.ndarray[float64, ndim=3] base1d = np.zeros((n_coor, 1, n_nod),
                                                       dtype=np.float64)
    if diff:
        bdim = dim

    else:
        bdim = 1

    cdef np.ndarray[float64, ndim=3] out = np.zeros((n_coor, bdim, n_nod),
                                                    dtype=np.float64)

    _f.array2fmfield3(_out, out)
    _f.fmf_pretend_nc(_bc, dim, 1, n_coor, 2, &bc[0, 0, 0])
    _f.array2fmfield2(_mtx_i, mtx_i)
    _f.array2fmfield3(_base1d, base1d)

    for ii in range(0, dim):
        _f.FMF_SetCell(_bc, ii)
        # slice [:,ii:ii+1]
        _f.fmf_pretend_nc(_coors, 1, 1, coors.shape[0], coors.shape[1],
                          &coors[0, ii])
        _get_barycentric_coors(_bc, _coors, _mtx_i, eps, check_errors)

    _eval_lagrange_tensor_product(_out, _bc, _mtx_i, _base1d,
                                  _nodes, nodes.shape[1], order, diff)

    return out

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef find_ref_coors(np.ndarray[float64, mode='c', ndim=2] ref_coors,
                     np.ndarray[int32, mode='c', ndim=2] cells,
                     np.ndarray[int32, mode='c', ndim=1] status,
                     np.ndarray[float64, mode='c', ndim=2] coors,
                     np.ndarray[int32, mode='c', ndim=1] ics,
                     np.ndarray[int32, mode='c', ndim=1] offsets,
                     np.ndarray[int32, mode='c', ndim=1] iconn,
                     np.ndarray[float64, mode='c', ndim=2] mesh_coors,
                     conns, eref_coorss, nodess,
                     mtx_is,
                     int allow_extrapolation,
                     float64 close_limit, float64 qp_eps,
                     int i_max, float64 newton_eps):
    """
    Find reference element coordinates corresponding to physical
    coordinates `coors`.

    This function works only with a geometry mesh (order 1
    connectivity), not a field mesh! The polynomial space arguments have
    to correspond to that.

    Returns
    -------
    ref_coors : array
        The reference coordinates.
    cells : array
        The array of (ig, iel) corresponding to the reference coordinates.
    status : array
        The status array: 0 is success, 1 means the point is
        extrapolated within `close_limit`, 2 extrapolated outside
        `close_limit` and 3 extrapolated with no extrapolation allowed.
    """
    cdef int32 ip, ic, ie, ig, iel, n_el, n_v, n_ep, n_ep_max, n_max, ok
    cdef int32 off, ie_min, n_col, ii, ik, order
    cdef int32 n_gr = len(conns)
    cdef int32 n_point = coors.shape[0]
    cdef int32 dim = coors.shape[1]
    cdef int32 *_nodes, *_cells = &cells[0, 0]
    cdef int32 *_status = &status[0]
    cdef float64 vmin, vmax, d_min, dist, aux
    cdef np.ndarray[int32, mode='c', ndim=2] conn
    cdef np.ndarray[int32, mode='c', ndim=2] nodes
    cdef FMField e_coors[1], _mesh_coors[1], xi[1], dest_point[1]
    cdef FMField eref_coors[1], _ref_coors[1],
    cdef FMField bc[1], base1d[1], mtx_i[1], bf[1]
    cdef float64 *buf_ec_max, *buf_b1d_max, *buf_bf_max
    cdef float64 buf6[6]

    # Prepare buffers.
    n_max = 0
    for ii in range(offsets.shape[0] - 1):
        n_max = Max_i(n_max, offsets[ii+1] - offsets[ii])

    n_ep_max = -1
    for ig in range(0, n_gr):
        conn = conns[ig]
        n_ep_max = Max_i(n_ep_max, conn.shape[1])

    buf_ec_max = <float64 *> pyalloc(n_ep_max * (dim + 2)
                                     * sizeof(float64))
    buf_b1d_max = buf_ec_max + n_ep_max * dim
    buf_bf_max = buf_b1d_max + n_ep_max

    _f.fmf_alloc(xi, n_max, 1, 1, dim)

    _f.array2fmfield2(_mesh_coors, mesh_coors)
    _f.array2fmfield2(_ref_coors, ref_coors)
    _f.fmf_pretend_nc(dest_point, n_point, 1, 1, dim, &coors[0, 0])

    _f.fmf_pretend_nc(_ref_coors, n_point, 1, 1, dim, &ref_coors[0, 0])

    # Point (destination coordinate) loop.
    for ip in range(0, n_point):
        ic = ics[ip]

        _f.FMF_SetCell(_ref_coors, ip)
        _f.FMF_SetCell(dest_point, ip)

        ok = 0
        d_min = 100.0 * 100.0

        n_el = offsets[ic+1] - offsets[ic]
        if n_el == 0:
            _status[ip] = 3
            _f.fmf_fillC(_ref_coors, 0.0)
            print 'standalone vertex!', ip, ic
            continue

        off = 2 * offsets[ic]

        # Containing element loop.
        for ie in range(0, n_el):
            ig = iconn[off + 2 * ie + 0]
            iel = iconn[off + 2 * ie + 1]

            nodes = nodess[ig]

            conn = conns[ig]
            if nodes.shape[0] != conn.shape[1]:
                msg = 'incompatible elements! (%d == %d)' % (nodes.shape[0],
                                                             conn.shape[1])
                raise ValueError(msg)

            # Finding reference coordinates uses geometric connectivity.
            _nodes = &nodes[0, 0]

            _f.FMF_SetCell(xi, ie)

            _f.array2fmfield2(eref_coors, eref_coorss[ig])
            _f.array2fmfield2(mtx_i, mtx_is[ig])

            n_v = eref_coors.nRow

            vmin = eref_coors.val[0]
            vmax = eref_coors.val[dim]

            n_ep = conn.shape[1]
            assert n_ep == n_v

            n_col = nodes.shape[1]

            _f.fmf_pretend_nc(e_coors, 1, 1, n_ep, dim, buf_ec_max)

            _f.ele_extractNodalValuesNBN(e_coors, _mesh_coors,
                                         &conn[0, 0] + n_ep * iel)

            if n_v == (dim + 1):
                _f.fmf_pretend_nc(bc, 1, 1, 1, dim + 1, buf6)
                _get_xi_simplex(xi, bc, dest_point, eref_coors, e_coors)
                # dist == 0 for 0 <= bc <= 1.
                dist = 0.0
                for ii in range(0, n_v):
                    aux = Min_f(Max_f(bc.val[ii] - 1.0, 0.0), 100.0)
                    dist += aux * aux
                    aux = Min_f(Max_f(0.0 - bc.val[ii], 0.0), 100.0)
                    dist += aux * aux

            else:
                _f.fmf_pretend_nc(base1d, 1, 1, 1, n_ep, buf_b1d_max)

                ok = _get_xi_tensor(xi, dest_point, e_coors,
                                    mtx_i, base1d, _nodes, n_col,
                                    vmin, vmax, i_max, newton_eps)
                # dist == 0 for vmin <= xi <= vmax and ok == 0.
                if ok == 0:
                    dist = 0.0
                    for ii in range(0, dim):
                        aux = Min_f(Max_f(xi.val[ii] - vmax, 0.0), 100.0)
                        dist += aux * aux
                        aux = Min_f(Max_f(vmin - xi.val[ii], 0.0), 100.0)
                        dist += aux * aux
                else:
                    dist = d_min + 1.0
                    ok = 0

            if dist < qp_eps:
                ok = 1
                ie_min = ie
                break

            else:
                if dist < d_min:
                    d_min = dist
                    ie_min = ie

        # Restore ig, iel.
        ig  = iconn[off + 2 * ie_min + 0]
        iel = iconn[off + 2 * ie_min + 1]

        # Store results
        _cells[2*ip+0] = ig
        _cells[2*ip+1] = iel

        if not ok:
            if allow_extrapolation:
                # Try using minimum distance xi.
                if sqrt(d_min) < close_limit:
                    _status[ip] = 1

                else:
                    _status[ip] = 2

                _f.FMF_SetCell(xi, ie_min)

            else:
                _status[ip] = 3

        else:
            _status[ip] = 0

        for ii in range(0, dim):
            _ref_coors.val[ii] =  xi.val[ii]

    pyfree(buf_ec_max)
    _f.fmf_free(xi)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef evaluate_in_rc(np.ndarray[float64, mode='c', ndim=2] out,
                     np.ndarray[float64, mode='c', ndim=2] ref_coors,
                     np.ndarray[int32, mode='c', ndim=2] cells,
                     np.ndarray[int32, mode='c', ndim=1] status,
                     np.ndarray[float64, mode='c', ndim=2] source_vals,
                     conns, eref_coorss, nodess,
                     np.ndarray[int32, mode='c', ndim=1] orders,
                     mtx_is, float64 qp_eps):
    """
    Evaluate source field DOF values in the given reference element
    coordinates using the given interpolation.

    1. Evaluate base functions in the reference coordinates.
    2. Interpolate source values using the base functions.

    Interpolation uses field approximation connectivity.
    """
    cdef int32 ip, ic, ig, iel, n_v, n_ep, n_ep_max
    cdef int32 n_col, ii, ik, order
    cdef int32 n_gr = len(conns)
    cdef int32 n_point = ref_coors.shape[0]
    cdef int32 dim = ref_coors.shape[1]
    cdef int32 dpn = out.shape[1]
    cdef int32 *_nodes, *_cells = &cells[0, 0]
    cdef int32 *_status = &status[0]
    cdef float64 vmin, vmax, aux
    cdef np.ndarray[int32, mode='c', ndim=2] conn
    cdef np.ndarray[int32, mode='c', ndim=2] nodes
    cdef FMField eref_coors[1], src[1]
    cdef FMField _ref_coors[1], _out[1], bc[1], base1d[1], mtx_i[1], bf[1]
    cdef FMField _source_vals[1]
    cdef float64 *buf_ec_max, *buf_b1d_max, *buf_bf_max, *buf_src_max
    cdef float64 buf6[6]

    # Prepare buffers.
    n_ep_max = -1
    for ig in range(0, n_gr):
        conn = conns[ig]
        n_ep_max = Max_i(n_ep_max, conn.shape[1])

    buf_ec_max = <float64 *> pyalloc(n_ep_max * (dim + 2 + dpn)
                                     * sizeof(float64))
    buf_b1d_max = buf_ec_max + n_ep_max * dim
    buf_bf_max = buf_b1d_max + n_ep_max
    buf_src_max = buf_bf_max + n_ep_max

    _f.array2fmfield2(_source_vals, source_vals)
    _f.fmf_pretend_nc(_out, n_point, 1, 1, dpn, &out[0, 0])
    _f.fmf_pretend_nc(_ref_coors, n_point, 1, 1, dim, &ref_coors[0, 0])

    # Point (destination coordinate) loop.
    for ip in range(0, n_point):
        _f.FMF_SetCell(_out, ip)
        _f.FMF_SetCell(_ref_coors, ip)

        if _status[ip] <= 1:
            ig = _cells[2*ip+0]
            iel = _cells[2*ip+1]

            nodes = nodess[ig]
            order = orders[ig]
            conn = conns[ig]
            _f.array2fmfield2(mtx_i, mtx_is[ig])

            _f.array2fmfield2(eref_coors, eref_coorss[ig])
            n_v = eref_coors.nRow

            n_ep = conn.shape[1]
            n_col = nodes.shape[1]

            _nodes = &nodes[0, 0]

            _f.fmf_pretend_nc(bf, 1, 1, 1, n_ep, buf_bf_max)

            if n_v == (dim + 1):
                _f.fmf_pretend_nc(bc, 1, 1, 1, dim + 1, buf6)
                _get_barycentric_coors(bc, _ref_coors, mtx_i, qp_eps, 0)
                _eval_lagrange_simplex(bf, bc, mtx_i,
                                       _nodes, n_col, order, 0)

            else:
                vmin = eref_coors.val[0]
                vmax = eref_coors.val[dim]

                _f.fmf_pretend_nc(bc, dim, 1, 1, 2, buf6)
                _f.fmf_pretend_nc(base1d, 1, 1, 1, n_ep, buf_b1d_max)

                for ii in range(0, dim):
                    _f.FMF_SetCell(bc, ii)
                    # slice [:,ii:ii+1]
                    bc.val[1] = (_ref_coors.val[ii] - vmin) / (vmax - vmin)
                    bc.val[0] = 1.0 - bc.val[1]
                _eval_lagrange_tensor_product(bf, bc, mtx_i, base1d,
                                              _nodes, n_col, order, 0)

            # Interpolate source_vals using bf.
            _f.fmf_pretend_nc(src, 1, 1, dpn, n_ep, buf_src_max)
            _f.ele_extractNodalValuesDBD(src, _source_vals,
                                         &conn[0, 0] + n_ep * iel)

            for ic in range(0, dpn):
                aux = 0.0
                for ik in range(0, n_ep):
                    aux += bf.val[ik] * src.val[n_ep*ic+ik]

                _out.val[ic] = aux

        else:
            _f.fmf_fillC(_out, 0.0)

    pyfree(buf_ec_max)
