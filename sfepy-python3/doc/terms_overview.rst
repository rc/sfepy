.. _term_overview:

Term Overview
=============

Term Syntax
-----------

In general, the syntax of a term call is:

.. centered::
   ``<term name>.<i>.<r>( <arg1>, <arg2>, ... )``,

where ``<i>`` denotes an integral name (i.e. a name of numerical quadrature
to use) and ``<r>`` marks a region (domain of the integral).

The following notation is used:

.. list-table:: Notation.
   :widths: 20 80
   :header-rows: 1

   * - symbol
     - meaning
   * - :math:`\Omega`
     - volume (sub)domain
   * - :math:`\Gamma`
     - surface (sub)domain
   * - :math:`d`
     - dimension of space
   * - :math:`t`
     - time
   * - :math:`y`
     - any function
   * - :math:`\ul{y}`
     - any vector function
   * - :math:`\ul{n}`
     - unit outward normal
   * - :math:`q`, :math:`s`
     - scalar test function
   * - :math:`p`, :math:`r`
     - scalar unknown or parameter function
   * - :math:`\bar{p}`
     - scalar parameter function
   * - :math:`\ul{v}`
     - vector test function
   * - :math:`\ul{w}`, :math:`\ul{u}`
     -  vector unknown or parameter function
   * - :math:`\ul{b}`
     - vector parameter function
   * - :math:`\ull{e}(\ul{u})`
     - Cauchy strain tensor (:math:`\frac{1}{2}((\nabla u) + (\nabla u)^T)`)
   * - :math:`\ull{F}`  
     - deformation gradient :math:`F_{ij} = \pdiff{x_i}{X_j}`
   * - :math:`J`
     - :math:`\det(F)`
   * - :math:`\ull{C}`
     -  right Cauchy-Green deformation tensor :math:`C = F^T F`
   * - :math:`\ull{E}(\ul{u})`
     - Green strain tensor :math:`E_{ij} = \frac{1}{2}(\pdiff{u_i}{x_j} +
       \pdiff{u_j}{x_i} + \pdiff{u_m}{x_i}\pdiff{u_m}{x_j})`
   * - :math:`\ull{S}`
     -  second Piola-Kirchhoff stress tensor
   * - :math:`\ul{f}`
     - vector volume forces
   * - :math:`f`
     - scalar volume force (source)
   * - :math:`\rho`
     - density
   * - :math:`\nu`
     - kinematic viscosity
   * - :math:`c`
     - any constant
   * - :math:`\delta_{ij}, \ull{I}`
     - Kronecker delta, identity matrix
   * - :math:`\tr{\ull{\bullet}}`
     - trace of a second order tensor (:math:`\sum_{i=1}^d \bullet_{ii}`)
   * - :math:`\dev{\ull{\bullet}}`
     - deviator of a second order tensor
       (:math:`\ull{\bullet} - \frac{1}{d}\tr{\ull{\bullet}}`)
   * - :math:`T_K \in \Tcal_h`
     - :math:`K`-th element of triangulation (= mesh) :math:`\Tcal_h` of
       domain :math:`\Omega`
   * - :math:`K \from \Ical_h`
     - :math:`K` is assigned values from :math:`\{0, 1, \dots, N_h-1\}
       \equiv \Ical_h` in ascending order

The suffix ":math:`_0`" denotes a quantity related to a previous time step.

Term names are (usually) prefixed according to the following conventions:

.. list-table:: Term name prefixes.
   :widths: 5 20 25 50
   :header-rows: 1

   * - prefix
     - meaning
     - evaluation modes
     - meaning
   * - dw
     - discrete weak
     - `'weak'`
     - terms having a virtual (test) argument and zero or more unknown
       arguments, used for FE assembling
   * - d
     - discrete
     - `'eval'`
     - terms having all arguments known, the result is the scalar value of
       the integral
   * - di
     - discrete integrated
     - `'eval'`
     - like 'd' but the result is not a scalar (e.g. a vector)
   * - dq
     - discrete quadrature
     - `'qp'`
     - terms having all arguments known, the result are the values in
       quadrature points of elements
   * - ev
     - evaluate
     - `'eval'`, `'el_avg'`, `'qp'`
     - terms having all arguments known and supporting all evaluation modes
       except `'weak'` (no virtual variables in arguments, no FE assembling)

.. _term_table:

Term Table
----------

Below we list all the terms available in an automatically generated table. The
first column lists the name, the second column the argument lists and the third
column the mathematical definition of each term.

The notation ``<virtual>`` corresponds to a test function,
``<state>`` to a unknown function and ``<parameter>`` to a known function. By
``<material>`` we denote material (constitutive) parameters, or, in general, any
given function of space and time that parameterizes a term, for example
a given traction force vector.

.. include:: term_table.rst
