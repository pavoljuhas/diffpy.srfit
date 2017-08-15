#!/usr/bin/env python
##############################################################################
#
# diffpy.srfit      by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2008 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Validatable class.

A Validatable has state that must be validated before a FitRecipe can first
calculate the residual.
"""

__all__ = ["Validatable"]


class Validatable(object):
    """Abstract class with state that must be validated by FitRecipe.
    """

    def _validateOthers(self, iterable):
        """Validate configuration of Validatable objects in the iterable.

        Parameters
        ----------
        iterable
            The objects to be validated.  Only the `Validatable` instances
            are checked, all other objects are ignored.

        Notes
        -----
            This is provided as a convenience for derived classes.
            Call this method from overloaded `_validate` with an
            iterable of objects to be validated.
        """
        for obj in iterable:
            if obj is self: continue
            if isinstance(obj, Validatable):
                obj._validate()
        return


    def _validate(self):
        """Validate self and then any other associated Validatables.

        This method must be overload in a derived class.

        Raises
        ------
        SrFitError
            If validation fails.
        """
        # Validate self in a derived class.
        # Then validate others.
        pass

# End class Validatable

# End of file
