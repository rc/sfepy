import numpy as nm

import sfepy.base.testing as tst

def cross2d(v1, v2):
    v1 = nm.asanyarray(v1)
    v2 = nm.asanyarray(v2)
    return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]

def test_make_cross_matrices():
    from sfepy.linalg import make_cross_matrices

    ok = True

    v1 = [[2.0, 3.0, -1.0],
          [1.0, 6.0, -1.0],
          [-4.0, 3.0, 3.0],
          [4.0, 3.0, 3.0]]
    v2 = v1[::-1]

    mtxs = make_cross_matrices(v1)
    cr = nm.einsum('cij,cj->ci', mtxs, v2)
    ncr = nm.cross(v1, v2)
    _ok = nm.allclose(cr, ncr, rtol=0.0, atol=1e-14)
    tst.report('3D make_cross_matrices(v1): %s' % _ok)
    ok = ok and _ok

    mtxs = make_cross_matrices(v1, first=False)
    cr = nm.einsum('cij,cj->ci', mtxs, v2)
    ncr = nm.cross(v2, v1)
    _ok = nm.allclose(cr, ncr, rtol=0.0, atol=1e-14)
    tst.report('3D make_cross_matrices(v1, first=False): %s' % _ok)
    ok = ok and _ok

    v1 = [[2.0, 3.0],
          [1.0, -1.0],
          [-4.0, 3.0]]
    v2 = v1[::-1]

    mtxs = make_cross_matrices(v1)
    cr = nm.einsum('cj,cj->c', mtxs, v2)
    ncr = cross2d(v1, v2)
    _ok = nm.allclose(cr, ncr, rtol=0.0, atol=1e-14)
    tst.report('2D make_cross_matrices(v1): %s' % _ok)
    ok = ok and _ok

    mtxs = make_cross_matrices(v1, first=False)
    cr = nm.einsum('cj,cj->c', mtxs, v2)
    ncr = cross2d(v2, v1)
    _ok = nm.allclose(cr, ncr, rtol=0.0, atol=1e-14)
    tst.report('2D make_cross_matrices(v1, first=False): %s' % _ok)
    ok = ok and _ok

    assert ok

def test_tensors():
    from sfepy.linalg import dot_sequences, insert_strided_axis

    ok = True

    a = nm.arange(1, 10).reshape(3, 3)
    b = nm.arange(9, 0, -1).reshape(3, 3)

    dab = nm.dot(a, b)
    dabt = nm.dot(a, b.T)
    datb = nm.dot(a.T, b)
    datbt = nm.dot(a.T, b.T)

    sa = insert_strided_axis(a, 0, 10)
    sb = insert_strided_axis(b, 0, 10)

    dsab = dot_sequences(sa, sb, mode='AB')
    _ok = nm.allclose(dab[None, ...], dsab, rtol=0.0, atol=1e-14)
    tst.report('dot_sequences AB: %s' % _ok)
    ok = ok and _ok

    dsabt = dot_sequences(sa, sb, mode='ABT')
    _ok = nm.allclose(dabt[None, ...], dsabt, rtol=0.0, atol=1e-14)
    tst.report('dot_sequences ABT: %s' % _ok)
    ok = ok and _ok

    dsatb = dot_sequences(sa, sb, mode='ATB')
    _ok = nm.allclose(datb[None, ...], dsatb, rtol=0.0, atol=1e-14)
    tst.report('dot_sequences ATB: %s' % _ok)
    ok = ok and _ok

    dsatbt = dot_sequences(sa, sb, mode='ATBT')
    _ok = nm.allclose(datbt[None, ...], dsatbt, rtol=0.0, atol=1e-14)
    tst.report('dot_sequences ATBT: %s' % _ok)
    ok = ok and _ok

    assert ok

def test_unique_rows():
    from sfepy.linalg import unique_rows

    a = nm.arange(1, 10).reshape(3, 3)

    b = nm.r_[a, a]
    c = unique_rows(b)

    ok = (a == c).all()

    assert ok

def test_assemble1d():
    from sfepy.linalg import assemble1d

    a = nm.arange(5)
    b = nm.arange(2)

    assemble1d(b, [1, 1, 1, 1, 0, 0], a[[0, 2, 3, 4, 1, 1]])

    ok = (b == [2, 10]).all()

    assert ok

def test_geometry():
    from sfepy.linalg import get_face_areas

    a1 = get_face_areas([[0, 1, 2, 3]],
                        [[0, 0], [1, 0], [1, 1], [0, 1]])

    a2 = get_face_areas([[0, 1, 2, 3]],
                        [[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]])
    ok = nm.allclose([a1, a2], [1, 1], rtol=0, atol=1e-15)

    assert ok

def test_get_blocks_stats():
    from sfepy.linalg.utils import get_blocks_stats

    A = nm.eye(3)
    B = nm.full((3,2), 2)
    C = nm.full((1,3), 3)
    D = nm.full((1,2), 4)

    sr = [slice(0, 3), slice(3, 5)]
    sc = [slice(0, 3), slice(3, 4)]

    M = nm.block([[A, B], [C, D]])
    stats = get_blocks_stats(M, sr, sc)

    assert stats['shape'].tolist() == [[(3, 3), (3, 1)], [(1, 3), (1, 1)]]
    assert (stats['min'] == nm.array([[0., 2.], [3., 4.]])).all()
    assert (stats['max'] == nm.array([[1., 2.], [3., 4.]])).all()
