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


"""Definition of __version__, __date__, __gitsha__.
"""

from diffpy.srfit._version import get_versions

v = get_versions()

__version__ = v['version']
__date__ = ' '.join((v['date'][:10], v['date'][11:19], v['date'][-5:]))
__gitsha__ = v['full-revisionid']

del v, get_versions

# End of file
