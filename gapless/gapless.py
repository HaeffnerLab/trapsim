'''
Tool for general gapless plane approximation solutions
for planar traps.
'''

import numpy as np
from itertools import *
import matplotlib.pyplot as plt
import numdifftools as nd

class Electrode():

    def __init__(self, location, axes_permutation=0):
        '''
        location is a 2-element list of the form
        [ (xmin, xmax), (ymin, ymax) ]

        axes_permutation is an integer.
        0 (default): normal surface trap. z-axis lies along the plane
        of the trap
        1: trap is in the x-y plane. z axis is vertical
        '''

        xmin, xmax = location[0]
        ymin, ymax = location[1]

        (self.x1, self.y1) = (xmin, ymin)
        (self.x2, self.y2) = (xmax, ymax)
        self.sub_electrodes = [] # list containing extra electrodes are connected to the current one

        self.axes_permutation = axes_permutation

    def solid_angle(self, r):
        '''
        The solid angle for an arbitary rectangle oriented along the grid is calculated by
        Gotoh, et al, Nucl. Inst. Meth., 96, 3

        The solid angle is calculated from the current electrode, plus any additional electrodes
        that are electrically connected to the current electrode. This allows you to join electrodes
        on the trap, or to make more complicated electrode geometries than just rectangles.
        '''
        if self.axes_permutation == 0:
            xp = r[0]
            yp = r[2]
            zp = r[1]
        if self.axes_permutation == 1:
            xp, yp, zp = r
        term = lambda x,y: np.arctan(((x - xp)*(y - yp))/(zp*np.sqrt( (x - xp)**2 + (y - yp)**2 + zp**2 )))
        solid_angle = abs(term(self.x2, self.y2) - term(self.x1, self.y2) - term(self.x2, self.y1) + term(self.x1, self.y1))

        for elec in self.sub_electrodes:
            solid_angle += elec.solid_angle(r)
        
        return solid_angle

    def extend(self, locations):
        '''
        Extend the current electrode by a set of rectangular regions
        '''
        for l in locations:
            elec = Electrode(l, axes_permutation = self.axes_permutation)
            self.sub_eub_electrodes.append(elec)

    def compute_voltage(self, r):
        '''
        Compute voltage at the observation point due only to this electrode. (That is,
        all other electrodes are grounded.)
        Is just the voltage on the electrode times the solid angle (over 2pi).
        Also since the charge is e, the potential energy due to this potential is already in eV
        '''

        return (self.voltage/(2*np.pi))*self.solid_angle(r)

    def compute_electric_field(self, r):
        '''
        Calculate the electric field at the observation point, given the voltage on the electrode
        If voltage is set in Volts, field is in Volts/meter.
        E = -grad(Potential)
        '''
        xp, yp, zp = r
        grad = nd.Gradient( self.compute_voltage )
        return -1*grad(r)

    def compute_d_effective(self, r):
        '''
        Calculate the effective distance due to this electrode. This is defined
        as the parallel plate capacitor separation which gives the observed electric
        field for the given applied voltage. That is,
        Deff = V/E. Will be different in each direction so we return [deff_x, deff_y, deff_z]
        '''
        [Ex, Ey, Ez] = self.compute_electric_field(r)
        return [self.voltage/Ex, self.voltage/Ey, self.voltage/Ez] # in meters
        
    def set_voltage(self, v):
        self.voltage = v

    def expand_potential( self, r):
        '''
        Numerically expand the potential due to the electrode to second order as a taylor series
        around the obersvation point r = [x, y, z]

        self.taylor_dict is a dictionary containing the terms of the expansion. e.g.
        self.taylor_dict['x^2'] = (1/2)d^2\phi/dx^2
        '''
        # first set the voltage to 1V for this. Save the old voltage to restore at the end.
        try:
            old_voltage = self.voltage
            self.set_voltage(1.0)
        except:
            print "no old voltage set"

        self.taylor_dict = {}
        grad = nd.Gradient( self.compute_voltage)
        self.taylor_dict['x'] = grad(r)[0]
        self.taylor_dict['y'] = grad(r)[1]
        self.taylor_dict['z'] = grad(r)[2]

        '''
        Now compute the second derivatives
        '''

        hessian = nd.Hessian( self.compute_voltage )
        self.taylor_dict['x^2'] = 0.5*hessian(r)[0][0]
        self.taylor_dict['y^2'] = 0.5*hessian(r)[1][1]
        self.taylor_dict['z^2'] = 0.5*hessian(r)[2][2]
        self.taylor_dict['xy'] = hessian(r)[0][1]
        self.taylor_dict['xz'] = hessian(r)[0][2]
        self.taylor_dict['zy'] = hessian(r)[1][2]
        try:
            # now restore the old voltage
            self.set_voltage(old_voltage)
        except:
            print "no old voltage set"

    def expand_in_multipoles( self, r, r0 = 1):
        '''
        Obtain the multipole expansion for the potential due to the elctrode at the observation point.
        '''
        
        # first, make sure we have a taylor expansion of the potential
        self.expand_potential(r)

        self.multipole_dict = {}
        # multipoles
        self.multipole_dict['U1'] = (r0**2)*(2*self.taylor_dict['x^2'] + self.taylor_dict['z^2'])
        self.multipole_dict['U2'] = (r0**2)*self.taylor_dict['z^2']
        self.multipole_dict['U3'] = 2*(r0**2)*self.taylor_dict['xy']
        self.multipole_dict['U4'] = 2*(r0**2)*self.taylor_dict['zy']
        self.multipole_dict['U5'] = 2*(r0**2)*self.taylor_dict['xz']

        # fields
        self.multipole_dict['Ex'] = -1*r0*self.taylor_dict['x']
        self.multipole_dict['Ey'] = -1*r0*self.taylor_dict['y']
        self.multipole_dict['Ez'] = -1*r0*self.taylor_dict['z']

class World():
    '''
    Compute all electrodes
    '''
    def __init__(self, axes_permutation = 0):
        self.electrode_dict = {}
        self.rf_electrode_dict = {}
        self.dc_electrode_dict = {}

    def add_electrode(self, name, xr, yr, kind, voltage = None):
        '''
        Add an electrode to the World. Optionally set a voltage on it. Name it with a string.
        kind = 'rf' or 'dc'. If kind == 'rf', then add this electrode to the rf electrode dict
        as well as to the general electrode dict
        '''
        e = Electrode([xr, yr])
        if voltage is not None:
            e.set_voltage(voltage)
        self.electrode_dict[name] = e
        
        if kind=='rf':
            self.rf_electrode_dict[name] = e
        if kind=='dc':
            self.dc_electrode_dict[name] = e

    def set_voltage(self, name, voltage):
        self.electrode_dict[name].set_voltage(voltage)

    def set_omega_rf(self, omega_rf):
        self.omega_rf = omega_rf

    def compute_total_rf_voltage(self, r):
        v = 0
        for e in self.rf_electrode_dict.keys():
            v += self.compute_voltage(e, r)
        return v

    def compute_total_dc_potential(self, r):
        v = 0
        for e in self.dc_electrode_dict.keys():
            v += e.compute_voltage(r)
        return v # the potential energy is automatically electron volts

    def compute_rf_field(self, r):

        '''
        Just add the electric field due to all the rf electrodes
        '''
        Ex = 0
        Ey = 0
        Ez = 0
        for name in self.rf_electrode_dict.keys():
            [ex, ey, ez] = self.rf_electrode_dict[name].compute_electric_field(r)
            Ex += ex
            Ey += ey
            Ez += ez
        return [Ex, Ey, Ez]

    def compute_squared_field_amplitude(self, r):
        Ex, Ey, Ez = self.compute_rf_field(r)
        
        return Ex**2 + Ey**2 + Ez**2 # has units of V^2/m^2

    def compute_pseudopot(self, r):
        q = 1.60217657e-19 # electron charge
        m = 6.64215568e-26 # 40 amu in kg
        omega_rf = self.omega_rf
        joule_to_ev = 6.24150934e18 # conversion factor to take joules -> eV
        E2 = self.compute_squared_field_amplitude(r)
        return (q**2/(4*m*omega_rf**2))*E2*joule_to_ev # psuedopotential in eV

    def compute_pseudopot_frequencies(self, r):
        '''
        This is only valid if xp, yp, zp is the trapping position. Return frequency (i.e. 2*pi*omega)
        '''
        ev_to_joule = 1.60217657e-19
        m = 6.64215568e-26 # 40 amu in kg
        hessdiag = nd.Hessdiag( self.compute_pseudopot )(r)
        d2Udx2 = ev_to_joule*hessdiag[0]
        d2Udy2 = ev_to_joule*hessdiag[1]
        d2Udz2 = ev_to_joule*hessdiag[2]
        '''
        Now d2Udx2 has units of J/m^2. Then w = sqrt(d2Udx2/(mass)) has units of angular frequency
        '''
        
        fx = np.sqrt(abs(d2Udx2)/m)/(2*np.pi)
        fy = np.sqrt(abs(d2Udy2)/m)/(2*np.pi)
        fz = np.sqrt(abs(d2Udz2)/m)/(2*np.pi)
        return [fx, fy, fz]

    def compute_dc_potential_frequencies(self, r):
        '''
        As always, this is valid only at the trapping position. Return frequency (not angular frequency)
        '''
        joule_to_ev = 6.24150934e18 # conversion factor to take joules -> eV
        m = 6.64215568e-26 # 40 amu in kg
        
        hessdiag = nd.Hessdiag( self.compute_dc_voltage )(r)
        d2Udx2 = ev_to_joule*hessdiag[0]
        d2Udy2 = ev_to_joule*hessdiag[1]
        d2Udz2 = ev_to_joule*hessdiag[2]
        fx = np.sqrt(abs(d2Udx2)/m)/(2*np.pi)
        fy = np.sqrt(abs(d2Udy2)/m)/(2*np.pi)
        fz = np.sqrt(abs(d2Udz2)/m)/(2*np.pi)
        return [fx, fy, fz]
