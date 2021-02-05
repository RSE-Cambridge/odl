"""Siemens geometry classes."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.tomo.util import axis_rotation_matrix, is_inside_bounds
from odl.tomo.util.utility import transform_system
from odl.tomo.geometry.geometry import AxisOrientedGeometry, Geometry
from odl.tomo.geometry.detector import Flat2dDetector
from odl.util import array_str, indent, signature_string


__all__ = ('BookletsGeometry', 'TiltedBookletsGeometry')

class BookletsGeometry(Geometry):

    """Abstract booklets beam geometry class.

    A geometry characterized by the presence of a segment-like ray source.
    """

    def src_position(self, angle, dparam):
        """Source position function.

        Parameters
        ----------
        angle : `array-like` or sequence
            Motion parameter(s) at which to evaluate. If
            ``motion_params.ndim >= 2``, a sequence of that length must be
            provided.
        dparam : `array-like` or sequence
            Detector parameter(s) at which to evaluate. If
            ``det_params.ndim >= 2``, a sequence of that length must be
            provided.

        Returns
        -------
        pos : `numpy.ndarray`
            Vector(s) pointing from the origin to the source point.
        """
        raise NotImplementedError('abstract method')

    def det_to_src(self, angle, dparam=(0,0), normalized=True):
        """Vector or direction from a detector location to the
        corresponding source point.

        The unnormalized version of this vector is computed as follows::

            vec = src_position(angle, dparam) -
                  det_point_position(angle, dparam)

        Parameters
        ----------
        angle : `array-like` or sequence
            One or several (Euler) angles in radians at which to
            evaluate. If ``motion_params.ndim >= 2``, a sequence of that
            length must be provided.
        dparam : `array-like` or sequence
            Detector parameter(s) at which to evaluate. If
            ``det_params.ndim >= 2``, a sequence of that length must be
            provided.

        Returns
        -------
        det_to_src : `numpy.ndarray`
            Vector(s) pointing from a detector point to the source (at
            infinity).
            The shape of the returned array is obtained from the
            (broadcast) shapes of ``angle`` and ``dparam``, and
            broadcasting is supported within both parameters and between
            them. The precise definition of the shape is
            ``broadcast(bcast_angle, bcast_dparam).shape + (ndim,)``,
            where ``bcast_angle`` is

            - ``angle`` if `motion_params` is 1D,
            - ``broadcast(*angle)`` otherwise,

            and ``bcast_dparam`` defined analogously.
        """
        # Always call the downstream methods with vectorized arguments
        # to be able to reliably manipulate the final axes of the result
        if self.motion_params.ndim == 1:
            squeeze_angle = (np.shape(angle) == ())
            angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        else:
            squeeze_angle = (np.broadcast(*angle).shape == ())
            angle = tuple(np.array(a, dtype=float, copy=False, ndmin=1)
                          for a in angle)

        if self.det_params.ndim == 1:
            squeeze_dparam = (np.shape(dparam) == ())
            dparam = np.array(dparam, dtype=float, copy=False, ndmin=1)
        else:
            squeeze_dparam = (np.broadcast(*dparam).shape == ())
            dparam = tuple(np.array(p, dtype=float, copy=False, ndmin=1)
                           for p in dparam)

        det_to_src = (self.src_position(angle, dparam) -
                      self.det_point_position(angle, dparam))

        if normalized:
            det_to_src /= np.linalg.norm(det_to_src, axis=-1, keepdims=True)

        if squeeze_angle and squeeze_dparam:
            det_to_src = det_to_src.squeeze()

        return det_to_src


class TiltedBookletsGeometry(BookletsGeometry, AxisOrientedGeometry):
    _default_config = dict(axis=(0, 0, 1),
                           src_to_det_init=(0, 1, 0),
                           det_axes_init=((1, 0, 0), (0, 0, 1)),
                           src_axis_init=(1, 0, 0))
    # TODO: remove unused attributes (det_curvature_radius, src_shift_func, det_shift_func)
    def __init__(self, apart, dpart, src_radius, det_radius,
                 det_curvature_radius=None, pitch=0, axis=(0, 0, 1),
                 src_shift_func=None, det_shift_func=None, **kwargs):

        default_axis = self._default_config['axis']
        default_src_to_det_init = self._default_config['src_to_det_init']
        default_det_axes_init = self._default_config['det_axes_init']

        # Handle initial coordinate system. We need to assign `None` to
        # the vectors first since we want to check that `init_matrix`
        # is not used together with those other parameters.
        src_to_det_init = kwargs.pop('src_to_det_init', None)
        det_axes_init = kwargs.pop('det_axes_init', None)

        # Store some stuff for repr
        if src_to_det_init is not None:
            self._src_to_det_init_arg = np.asarray(src_to_det_init,
                                                   dtype=float)
        else:
            self._src_to_det_init_arg = None

        if det_axes_init is not None:
            self._det_axes_init_arg = tuple(
                np.asarray(a, dtype=float) for a in det_axes_init)
        else:
            self._det_axes_init_arg = None

        # Compute the transformed system and the transition matrix. We
        # transform only those vectors that were not explicitly given.
        vecs_to_transform = []
        if src_to_det_init is None:
            vecs_to_transform.append(default_src_to_det_init)
        if det_axes_init is None:
            vecs_to_transform.extend(default_det_axes_init)

        transformed_vecs = transform_system(
            axis, default_axis, vecs_to_transform)
        transformed_vecs = list(transformed_vecs)

        axis = transformed_vecs.pop(0)
        if src_to_det_init is None:
            src_to_det_init = transformed_vecs.pop(0)
        if det_axes_init is None:
            det_axes_init = (transformed_vecs.pop(0), transformed_vecs.pop(0))
        assert transformed_vecs == []

        # Check and normalize `src_to_det_init`. Detector axes are
        # normalized in the detector class.
        if np.linalg.norm(src_to_det_init) == 0:
            raise ValueError('`src_to_det_init` cannot be zero')
        else:
            src_to_det_init /= np.linalg.norm(src_to_det_init)

        # Get stuff out of kwargs, otherwise upstream code complains
        # about unknown parameters (rightly so)
        self.__pitch = float(pitch)
        self.__offset_along_axis = float(kwargs.pop('offset_along_axis', 0))
        self.__src_radius = float(src_radius)

        # Initialize stuff
        self.__src_to_det_init = src_to_det_init
        AxisOrientedGeometry.__init__(self, axis)

        check_bounds = kwargs.get('check_bounds', True)
        
        detector = Flat2dDetector(dpart, axes=det_axes_init,
                                  check_bounds=check_bounds)

        super(TiltedBookletsGeometry, self).__init__(
            ndim=3, motion_part=apart, detector=detector, **kwargs)

        # Check parameters
        if self.src_radius < 0:
            raise ValueError('source circle radius {} is negative'
                             ''.format(src_radius))
        self.__det_radius = float(det_radius)
        if self.det_radius < 0:
            raise ValueError('detector circle radius {} is negative'
                             ''.format(det_radius))

        if self.src_radius == 0 and self.det_radius == 0:
            raise ValueError('source and detector circle radii cannot both be '
                             '0')

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` has dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))
        
        # TODO: remove these attributes or implement them correctly!

        self.src_shift_func = lambda x: np.array(
                [0.0, 0.0, 0.0], dtype=float, ndmin=2)
        self.det_shift_func = lambda x: np.array(
                [0.0, 0.0, 0.0], dtype=float, ndmin=2)

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self.__src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self.__det_radius

    @property
    def pitch(self):
        """Constant vertical distance traversed in a full rotation."""
        return self.__pitch

    @property
    def src_to_det_init(self):
        """Initial state of the vector pointing from source to detector
        reference point."""
        return self.__src_to_det_init

    @property
    def det_axes_init(self):
        """Initial axes defining the detector orientation."""
        return self.detector.axes

    @property
    def offset_along_axis(self):
        """Scalar offset along ``axis`` at ``angle=0``."""
        return self.__offset_along_axis

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]


    def det_axes(self, angle):
        # Transpose to take dot along axis 1
        axes = self.rotation_matrix(angle).dot(self.det_axes_init.T)
        # `axes` has shape (a, 3, 2), need to roll the last dimensions
        # to the second-to-last place
        return np.rollaxis(axes, -1, -2)


    def det_refpoint(self, angle):
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        rot_matrix = self.rotation_matrix(angle)
        extra_dims = angle.ndim
        det_shifts = np.array(self.det_shift_func(angle), dtype=float, ndmin=2)

        # Initial vector from center of rotation to detector.
        # It can be computed this way since source and detector are at
        # maximum distance, i.e. the connecting line passes the center.
        center_to_det_init = self.det_radius * self.src_to_det_init
        # shifting the detector according to det_shift_func
        tangent = -np.cross(self.src_to_det_init, self.axis)
        tangent /= np.linalg.norm(tangent)
        det_shift = (np.multiply.outer(det_shifts[:, 0], self.src_to_det_init)
                     + np.multiply.outer(det_shifts[:, 1], tangent))
        center_to_det_init = center_to_det_init + det_shift
        # `circle_component` has shape (a, ndim)
        circle_component = np.einsum('...ij,...j->...i',
                                     rot_matrix, center_to_det_init)

        # Increment along the rotation axis according to pitch and
        # offset_along_axis
        # `shift_along_axis` has shape angles.shape
        shift_along_axis = (self.offset_along_axis
                            + self.pitch * angle / (2 * np.pi)
                            + det_shifts[:, 2])
        # Create outer product of `shift_along_axis` and `axis`, resulting
        # in shape (a, ndim)
        pitch_component = np.multiply.outer(shift_along_axis, self.axis)

        # Broadcast translation along extra dimensions
        transl_slc = (None,) * extra_dims + (slice(None),)
        refpt = (self.translation[transl_slc]
                 + circle_component
                 + pitch_component)
        if squeeze_out:
            refpt = refpt.squeeze()

        return refpt
    
    def src_axis(self, angle):
        axes = self.det_axes(angle)
        return axes[...,0,:]

    def src_position(self, angle, dparam):
        """Return the source point at ``angle`` corresponding to a
        detector location.

        For an angle ``phi`` and detector parameters ``dparams``,
        the source position is given by ::

            src(phi, dparam) = translation +
                       rot_matrix(phi) * (-src_rad * src_to_det_init) +
                       (offset_along_axis + pitch * phi) * axis +
                       source_shift(phi) -
                       src_axis(phi) * dparam[0]

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector and
            source_shift(phi) = rot_matrix(phi) *
                                (shift1 * (-src_to_det_init) +
                                shift2 * cross(src_to_det_init, axis))
                                shift3 * axis

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of the detector.
        
        dparam : `array-like` or sequence of lenght 2 of
            Detector parameter(s) at which to evaluate.

        Returns
        -------
        pos : `numpy.ndarray`, shape (3,) or (num_angles, 3)
            Vector(s) pointing from the origin to the source position.
            If ``angle`` is a single parameter, the returned array has
            shape ``(3,)``, otherwise ``angle.shape + (3,)``.
        """
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        rot_matrix = self.rotation_matrix(angle)
        extra_dims = angle.ndim
        src_shifts = self.src_shift_func(angle)

        # Initial vector from center of rotation to source.
        # It can be computed this way since source and detector are at
        # maximum distance, i.e. the connecting line passes the center.
        center_to_src_init = -self.src_radius * self.src_to_det_init
        # shifting the source according to ffs
        tangent = -np.cross(-self.src_to_det_init, self.axis)
        tangent /= np.linalg.norm(tangent)
        ffs_shift = (np.multiply.outer(src_shifts[:, 0],
                                       -self.src_to_det_init)
                     + np.multiply.outer(src_shifts[:, 1], tangent))
        center_to_src_init = center_to_src_init + ffs_shift
        circle_component = np.einsum('...ij,...j->...i',
                                     rot_matrix, center_to_src_init)


        # Increment along the rotation axis according to pitch and
        # offset_along_axis
        # `shift_along_axis` has shape angles.shape
        shift_along_axis = (self.offset_along_axis
                            + self.pitch * angle / (2 * np.pi)
                            + src_shifts[:, 2])

        # Create outer product of `shift_along_axis` and `axis`, resulting
        # in shape (a, ndim)
        pitch_component = np.multiply.outer(shift_along_axis, self.axis)

        # Increment along the source axis according to dparam[0]
        ## TODO rm comments and check dparam shape!!!!
        sparam = np.array(dparam[0], dtype=float, copy=False, ndmin=1)
        print("sparam", sparam.shape)
        source_offset = sparam[:, None] * self.src_axis(angle)
        print("self.src_axis(angle)", self.src_axis(angle).shape)
        # source_offset = np.einsum('...ij,...j->...i',
        #                              sparam, self.src_axis(angle))
        # source_offset = np.multiply.outer(sparam, self.src_axis(angle))
        # print("source_offset", source_offset.shape)

        # Broadcast translation along extra dimensions
        transl_slc = (None,) * extra_dims + (slice(None),)

        # TODO: rm comments
        # refpt = (self.translation[transl_slc]
        #          + circle_component
        #          + pitch_component)
        # print("refpt.shape",refpt.shape)
        refpt = (self.translation[transl_slc]
                 + circle_component
                 + pitch_component
                 + source_offset)
        if squeeze_out:
            refpt = refpt.squeeze()

        return refpt

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = [('src_radius', self.src_radius, -1),
                   ('det_radius', self.det_radius, -1),
                   ('pitch', self.pitch, 0)
                   ]

        if not np.allclose(self.axis, self._default_config['axis']):
            optargs.append(['axis', array_str(self.axis), ''])

        optargs.append(['offset_along_axis', self.offset_along_axis, 0])

        if self._src_to_det_init_arg is not None:
            optargs.append(['src_to_det_init',
                            array_str(self._src_to_det_init_arg),
                            None])

        if self._det_axes_init_arg is not None:
            optargs.append(
                ['det_axes_init',
                 tuple(array_str(a) for a in self._det_axes_init_arg),
                 None])

        if not np.array_equal(self.translation, (0, 0, 0)):
            optargs.append(['translation', array_str(self.translation), ''])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(sig_str))


    def __getitem__(self, indices):
        """Return self[indices].

        This is defined by ::

            self[indices].partition == self.partition[indices]

        where all other parameters are the same.

        Examples
        --------
        >>> apart = odl.uniform_partition(0, 4, 4)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], [20, 20])
        >>> geom = odl.tomo.ConeBeamGeometry(apart, dpart, 50, 100, pitch=2)

        Extract sub-geometry with every second angle:

        >>> geom[::2]
        ConeBeamGeometry(
            nonuniform_partition(
                [ 0.5,  2.5],
                min_pt=0.0, max_pt=4.0
            ),
            uniform_partition([-1., -1.], [ 1.,  1.], (20, 20)),
            src_radius=50.0,
            det_radius=100.0,
            pitch=2.0
        )
        """
        part = self.partition[indices]
        apart = part.byaxis[0]
        dpart = part.byaxis[1:]

        return TiltedBookletsGeometry(apart, dpart,
                                src_radius=self.src_radius,
                                det_radius=self.det_radius,
                                det_curvature_radius=self.det_curvature_radius,
                                pitch=self.pitch,
                                axis=self.axis,
                                offset_along_axis=self.offset_along_axis,
                                src_to_det_init=self._src_to_det_init_arg,
                                det_axes_init=self._det_axes_init_arg,
                                src_shift_func=self.src_shift_func,
                                det_shift_func=self.det_shift_func,
                                translation=self.translation)

    # Manually override the abstract method in `Geometry` since it's found
    # first
    rotation_matrix = AxisOrientedGeometry.rotation_matrix