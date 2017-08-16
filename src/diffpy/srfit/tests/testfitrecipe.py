#!/usr/bin/env python
##############################################################################
#
# diffpy.srfit      by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Tests for refinableobj module."""

import unittest

from numpy import linspace, array_equal, pi, sin, dot

from diffpy.srfit.fitbase.fitrecipe import FitRecipe
from diffpy.srfit.fitbase.fitcontribution import FitContribution
from diffpy.srfit.fitbase.profile import Profile
from diffpy.srfit.fitbase.parameter import Parameter
from diffpy.srfit.fitbase import Calculator, ProfileGenerator
from diffpy.srfit.exceptions import SrFitError


class TestFitRecipe(unittest.TestCase):

    def setUp(self):
        self.recipe = FitRecipe("recipe")
        self.recipe.fithooks[0].verbose = 0

        # Set up the Profile
        self.profile = Profile()
        x = linspace(0, pi, 10)
        y = sin(x)
        self.profile.setObservedProfile(x, y)

        # Set up the FitContribution
        self.fitcontribution = FitContribution("cont")
        self.fitcontribution.setProfile(self.profile)
        self.fitcontribution.setEquation("A*sin(k*x + c)")
        self.fitcontribution.A.setValue(1)
        self.fitcontribution.k.setValue(1)
        self.fitcontribution.c.setValue(0)

        self.recipe.addContribution(self.fitcontribution)
        return

    def testFixFree(self):
        recipe = self.recipe
        con = self.fitcontribution

        recipe.addVar(con.A, 2, tag = "tagA")
        recipe.addVar(con.k, 1, tag = "tagk")
        recipe.addVar(con.c, 0)
        recipe.newVar("B", 0)

        self.assertTrue(recipe.isFree(recipe.A))
        recipe.fix("tagA")
        self.assertFalse(recipe.isFree(recipe.A))
        recipe.free("tagA")
        self.assertTrue(recipe.isFree(recipe.A))
        recipe.fix("A")
        self.assertFalse(recipe.isFree(recipe.A))
        recipe.free("A")
        self.assertTrue(recipe.isFree(recipe.A))
        recipe.fix(recipe.A)
        self.assertFalse(recipe.isFree(recipe.A))
        recipe.free(recipe.A)
        self.assertTrue(recipe.isFree(recipe.A))
        recipe.fix(recipe.A)
        self.assertFalse(recipe.isFree(recipe.A))
        recipe.free("all")
        self.assertTrue(recipe.isFree(recipe.A))
        self.assertTrue(recipe.isFree(recipe.k))
        self.assertTrue(recipe.isFree(recipe.c))
        self.assertTrue(recipe.isFree(recipe.B))
        recipe.fix(recipe.A, "tagk", c = 3)
        self.assertFalse(recipe.isFree(recipe.A))
        self.assertFalse(recipe.isFree(recipe.k))
        self.assertFalse(recipe.isFree(recipe.c))
        self.assertTrue(recipe.isFree(recipe.B))
        self.assertEqual(3, recipe.c.value)
        recipe.fix("all")
        self.assertFalse(recipe.isFree(recipe.A))
        self.assertFalse(recipe.isFree(recipe.k))
        self.assertFalse(recipe.isFree(recipe.c))
        self.assertFalse(recipe.isFree(recipe.B))

        self.assertRaises(ValueError, recipe.free, "junk")
        self.assertRaises(ValueError, recipe.fix, tagA = 1)
        self.assertRaises(ValueError, recipe.fix, "junk")
        return

    def testVars(self):
        """Test to see if variables are added and removed properly."""
        recipe = self.recipe
        con = self.fitcontribution

        recipe.addVar(con.A, 2)
        recipe.addVar(con.k, 1)
        recipe.addVar(con.c, 0)
        recipe.newVar("B", 0)

        names = recipe.getNames()
        self.assertEqual(names, ["A", "k", "c", "B"])
        values = recipe.getValues()
        self.assertTrue((values == [2, 1, 0, 0]).all())

        # Constrain a parameter to the B-variable to give it a value
        p = Parameter("Bpar", -1)
        recipe.constrain(recipe.B, p)
        values = recipe.getValues()
        self.assertTrue((values == [2, 1, 0]).all())
        recipe.delVar(recipe.B)

        recipe.fix(recipe.k)

        names = recipe.getNames()
        self.assertEqual(names, ["A", "c"])
        values = recipe.getValues()
        self.assertTrue((values == [2, 0]).all())

        recipe.fix("all")
        names = recipe.getNames()
        self.assertEqual(names, [])
        values = recipe.getValues()
        self.assertTrue((values == []).all())

        recipe.free("all")
        names = recipe.getNames()
        self.assertEqual(3, len(names))
        self.assertTrue("A" in names)
        self.assertTrue("k" in names)
        self.assertTrue("c" in names)
        values = recipe.getValues()
        self.assertEqual(3, len(values))
        self.assertTrue(0 in values)
        self.assertTrue(1 in values)
        self.assertTrue(2 in values)
        return


    def testResidual(self):
        """Test the residual and everything that can change it."""

        # With thing set up as they are, the residual should be 0
        res = self.recipe.residual()
        self.assertAlmostEqual(0, dot(res, res))

        # Change the c value to 1 so that the equation evaluates as sin(x+1)
        x = self.profile.x
        y = sin(x+1)
        self.recipe.cont.c.setValue(1)
        res = self.recipe.residual()
        self.assertTrue( array_equal(y-self.profile.y, res) )

        # Try some constraints
        # Make c = 2*A, A = Avar
        var = self.recipe.newVar("Avar")
        self.recipe.constrain(self.fitcontribution.c, "2*A",
                {"A" : self.fitcontribution.A})
        self.assertEqual(2, self.fitcontribution.c.value)
        self.recipe.constrain(self.fitcontribution.A, var)
        self.assertEqual(1, var.getValue())
        self.assertEqual(self.recipe.cont.A.getValue(), var.getValue())
        # c is constrained to a constrained parameter.
        self.assertEqual(2, self.fitcontribution.c.value)
        # The equation should evaluate to sin(x+2)
        x = self.profile.x
        y = sin(x+2)
        res = self.recipe.residual()
        self.assertTrue( array_equal(y-self.profile.y, res) )

        # Now try some restraints. We want c to be exactly zero. It should give
        # a penalty of (c-0)**2, which is 4 in this case
        r1 = self.recipe.restrain(self.fitcontribution.c, 0, 0, 1)
        self.recipe._ready = False
        res = self.recipe.residual()
        chi2 = 4 + dot(y - self.profile.y, y - self.profile.y)
        self.assertAlmostEqual(chi2, dot(res, res) )

        # Clear the constraint and restore the value of c to 0. This should
        # give us chi2 = 0 again.
        self.recipe.unconstrain(self.fitcontribution.c)
        self.fitcontribution.c.setValue(0)
        res = self.recipe.residual([self.recipe.cont.A.getValue()])
        chi2 = 0
        self.assertAlmostEqual(chi2, dot(res, res) )

        # Remove the restraint and variable
        self.recipe.unrestrain(r1)
        self.recipe.delVar(self.recipe.Avar)
        self.recipe._ready = False
        res = self.recipe.residual()
        chi2 = 0
        self.assertAlmostEqual(chi2, dot(res, res) )

        # Add constraints at the fitcontribution level.
        self.fitcontribution.constrain(self.fitcontribution.c, "2*A")
        # This should evaluate to sin(x+2)
        x = self.profile.x
        y = sin(x+2)
        res = self.recipe.residual()
        self.assertTrue( array_equal(y-self.profile.y, res) )

        # Add a restraint at the fitcontribution level.
        r1 = self.fitcontribution.restrain(self.fitcontribution.c, 0, 0, 1)
        self.recipe._ready = False
        # The chi2 is the same as above, plus 4
        res = self.recipe.residual()
        x = self.profile.x
        y = sin(x+2)
        chi2 = 4 + dot(y - self.profile.y, y - self.profile.y)
        self.assertAlmostEqual(chi2, dot(res, res) )

        # Remove those
        self.fitcontribution.unrestrain(r1)
        self.recipe._ready = False
        self.fitcontribution.unconstrain(self.fitcontribution.c)
        self.fitcontribution.c.setValue(0)
        res = self.recipe.residual()
        chi2 = 0
        self.assertAlmostEqual(chi2, dot(res, res) )

        # Now try to use the observed profile inside of the equation
        # Set the equation equal to the data
        self.fitcontribution.setEquation("y")
        res = self.recipe.residual()
        self.assertAlmostEqual(0, dot(res, res))

        # Now add the uncertainty. This should give dy/dy = 1 for the residual
        self.fitcontribution.setEquation("y+dy")
        res = self.recipe.residual()
        self.assertAlmostEqual(len(res), dot(res, res))

        return


    def testCalculatorValidation(self):
        """Verify Calculator object validates as expected.
        """
        class LineCalc(Calculator):

            def __init__(self, name):
                Calculator.__init__(self, name)
                self.newParameter("slope", 1.0)
                self.newParameter("intercept", 0.0)
                return

            def __call__(self, x):
                a = self.slope.getValue()
                b = self.intercept.getValue()
                return a * x + b
        # End of class LineCalc

        recipe = self.recipe
        cont = self.fitcontribution
        # use LineCalc as a model function in the contribution
        lc = LineCalc("line")
        cont.registerCalculator(lc)
        cont.setEquation('line')
        x0 = self.profile.xobs
        y0 = self.profile.yobs
        y1 = recipe.residual()
        # verify that "line" is used for model calculation
        self.assertTrue(array_equal(x0 - y0, y1))
        # verify "line" is invalid with a Parameter unset
        recipe._updateConfiguration()
        lc.intercept << None
        self.assertRaises(SrFitError, recipe.residual)
        cont.line.intercept << 0
        cont.line.slope << None
        recipe._updateConfiguration()
        self.assertRaises(SrFitError, recipe.residual)
        cont.line.slope << 0
        y2 = recipe.residual()
        self.assertTrue(array_equal(-y0, y2))
        return


    def testProfileValidation(self):
        """Verify Profile object validates as expected.
        """
        profile = self.profile
        recipe = self.recipe
        # initial profile should be valid
        recipe._updateConfiguration()
        recipe.residual()
        x0 = profile.x.copy()
        profile.x = linspace(0, 5, 6)
        recipe._updateConfiguration()
        self.assertRaises(SrFitError, recipe.residual)
        profile.x = x0
        recipe.residual()
        profile.dy = linspace(0, 5, 6)
        recipe._updateConfiguration()
        self.assertRaises(SrFitError, recipe.residual)
        return


    def testProfileGeneratorValidation(self):
        """Verify ProfileGenerator object validates as expected.
        """
        recipe = self.recipe
        pgen = ProfileGenerator('gen')
        recipe.cont.addProfileGenerator(pgen)
        # initial ProfileGenerator should be valid
        recipe.residual()
        # invalidate the ProfileGenerator
        pgen.profile = None
        recipe._updateConfiguration()
        self.assertRaises(SrFitError, recipe.residual)
        return

# End of class TestFitRecipe

if __name__ == "__main__":
    unittest.main()
