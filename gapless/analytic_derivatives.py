'''
Analytic derivatives of the solid angle of rectangles.
'''

from sympy import *

class analytic_derivatives():

    def __init__(self, axes_permutation=0):
        x, y, z, xp, yp, zp = symbols('x y z xp yp zp')

        if axes_permutation = 0:
            self.term = atan( ((x - xp)*(y - zp)) / (yp*sqrt( (x - xp)**2 + (y - zp)**2 + yp**2)))

        if axes_permutation = 1:
            self.term = atan( ((x - xp)*(y - yp)) / (zp*sqrt( (x - xp)**2 + (y - yp)**2 + zp**2 )) )

        self.symbols = [x,y,z,xp,yp,zp]

        self.functions_dict = {}

        self.first_order()
        self.second_order()
        self.third_order()

    def first_order(self):
        
        x, y, z, xp, yp, zp = self.symbols

        self.functions_dict['ddx'] = lambdify(self.symbols, self.term.diff(xp))
        self.functions_dict['ddy'] = lambdify(self.symbols, self.term.diff(yp))
        self.functions_dict['ddz'] = lambdify(self.symbols, self.term.diff(zp))

    def second_order(self):
        
        x, y, z, xp, yp, zp = self.symbols
        
        self.functions_dict['d2dx2'] = lambdify(self.symbols, self.term.diff(xp,2))
        self.functions_dict['d2dy2'] = lambdify(self.symbols, self.term.diff(yp,2))
        self.functions_dict['d2dz2'] = lambdify(self.symbols, self.term.diff(zp,2))
        self.functions_dict['d2dxdy'] = lambdify(self.symbols, self.term.diff(xp,yp))
        self.functions_dict['d2dxdz'] = lambdify(self.symbols, self.term.diff(xp,zp))
        self.functions_dict['d2dydz'] = lambdify(self.symbols, self.term.diff(yp,zp))

    def third_order(self):

        x, y, z, xp, yp, zp = self.symbols
        
        self.functions_dict['d3dx3'] = lambdify(self.symbols, self.term.diff(xp,3))
        self.functions_dict['d3dy3'] = lambdify(self.symbols, self.term.diff(yp,3))
        self.functions_dict['d3dz3'] = lambdify(self.symbols, self.term.diff(zp,3))

    def fourth_order(self):

        x, y, z, xp, yp, zp = self.symbols
        
        self.functions_dict['d4dx4'] = lambdify(self.symbols, self.term.diff(xp,4))
        self.functions_dict['d4dy4'] = lambdify(self.symbols, self.term.diff(yp,4))
        self.functions_dict['d4dz4'] = lambdify(self.symbols, self.term.diff(zp,4))
