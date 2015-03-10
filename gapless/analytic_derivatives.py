'''
Analytic derivatives of the solid angle of rectangles.
'''

from sympy import *

class analytic_derivatives():

    def __init__(self):
        x, y, z, xp, yp, zp = symbols('x y z xp yp zp')

        term = atan( ((x - xp)*(y - yp)) / (zp*sqrt( (x - xp)**2 + (y - yp)**2 + zp**2 )) )

        


class analytic_derivatives():

    def first_order(self, var, r, x, y):
        xp, yp, zp = r
        dx = x - xp
        dy = y - yp
        r0 = np.sqrt( dx**2 + dy**2 + zp**2)
        if var == 'x':
            return (-dy*zp)/ (( dx**2 + zp**2)*r0)
        if var == 'y':
            return (-dx*zp)/ ((dy**2 + zp**2)*r0)
        if var == 'z':
            return -dx*dy*( dx**2 + dy**2 + 2*zp**2) / ((dx**2 + zp**2)*(dy**2 + zp**2)*r0)

    def second_order(self, var, r, x, y):
        
        xp, yp, zp = r
        dx = x - xp
        dy = y - yp
        r02 = dx**2 + dy**2 + zp**2
        r032 = (dx**2 + dy**2 + zp**2)**(3./2.)
        r052 = (dx**2 + dy**2 + zp**2)**(5./2.)

        if var == 'x^2':
            num =  -dx*dy*zp*(3*dx**2 + 2*dy**2 + 3*zp**2)
            denom = (dx**2 + zp**2)**2 * r032
            return num/denom

        if var == 'y^2':
            num = -dx*dy*zp*(2*dx**2 + 3*dy**2 + 3*zp**2)
            denom = (dy**2 + zp**2)**2*r032
            return num/denom

        if var == 'z^2':
            t1 = -2*(dx*dy*zp)**2*r02*(dx**2 + dy**2 + 2 zp**2)**2 /  (dx**2 + zp**2)**2 / ( dy**2 + zp**2)**2
            smallFrac = dx**2*dy**2/ zp**2 / r02
            t2 = (2*(dx**2 + dy**2)**2 + 5( dx**2 + dy**2)*zp**2 + 6*zp**4)/(1 + smallFrac)
            return dx*dy*(t1 + t2) / zp**3 / r052
        
        if var == 'xy':
            return zp/r032

        if var == 'xz':
            num = dy*( -dx**2*(dx**2 + dy**2) + ( dx**2 + dy**2 )*zp**2 + 2*zp**4)
            denom = (dx**2 = zp**2)*r032
            return num/denom

        if var == 'yz':
            num = dx*( -(dx**2 + dy**2)*dy**2 + (dx**2 + dy**2)*zp**2 + 2*zp**4)
            denom = (dy**2 + zp**2)**2*r032
            return num/denom            
