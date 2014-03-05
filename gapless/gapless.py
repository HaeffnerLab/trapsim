'''
Tool for general gapless plane approximation solutions
for planar traps.
'''

import numpy as np
from itertools import *
import matplotlib.pyplot as plt

class Electrode():

    def __init__(self, location):
        '''
        location is a 2-element list of the form
        [ (xmin, xmax), (ymin, ymax) ]
        '''

        xmin, xmax = location[0]
        ymin, ymax = location[1]

        (self.x1, self.y1) = (xmin, ymin)
        (self.x2, self.y2) = (xmax, ymax)

    def solid_angle(self, xp, yp, zp):
        '''
        The solid angle for an arbitary rectangle oriented along the grid is calculated by
        Gotoh, et al, Nucl. Inst. Meth., 96, 3
        '''
        term = lambda x,y: np.arctan(((x - xp)*(y - yp))/(zp*np.sqrt( (x - xp)**2 + (y - yp)**2 + zp**2 )))
        return abs(term(self.x2, self.y2) - term(self.x1, self.y2) - term(self.x2, self.y1) + term(self.x1, self.y1))

    def set_voltage(self, v):
        self.voltage = v

class World():
    '''
    Compute all electrodes
    '''
    def __init__(self, omega_rf, d=10e-9):
        self.electrode_dict = {}
        self.rf_electrode_dict = {}
        self.dc_electrode_dict = {}
        self.d = d # step-size for derivative
        self.omega_rf = omega_rf

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

    def compute_voltage(self, name, xp, yp, zp):
        e = self.electrode_dict[name]
        v = e.voltage
        omega = e.solid_angle(xp, yp, zp)
        return (v/(2*np.pi))*omega

    def compute_total_rf_voltage(self, xp, yp, zp):
        v = 0
        for e in self.rf_electrode_dict.keys():
            v += self.compute_voltage(e, xp, yp, zp)
        return v

    def compute_total_dc_voltage(self, xp, yp, zp):
        v = 0
        for e in self.dc_electrode_dict.keys():
            v += self.compute_voltage(e, xp, yp, zp)
        return v # the potential energy is automatically electron volts

    def compute_rf_field(self, xp, yp, zp):

        '''
        If voltage is set in Volts, field is in volts/meter
        '''
        d = self.d
        base_voltage = self.compute_total_rf_voltage(xp, yp, zp)
        Ex = -(self.compute_total_rf_voltage(xp+d, yp, zp) - base_voltage)/d
        Ey = -(self.compute_total_rf_voltage(xp, yp+d, zp) - base_voltage)/d
        Ez = -(self.compute_total_rf_voltage(xp, yp, zp+d) - base_voltage)/d

        return [Ex, Ey, Ez]

    def compute_squared_field_amplitude(self, xp, yp, zp):
        Ex, Ey, Ez = self.compute_rf_field(xp,yp,zp)
        
        return Ex**2 + Ey**2 + Ez**2 # has units of V^2/m^2

    def compute_pseudopot(self, xp, yp, zp):
        q = 1.60217657e-19 # electron charge
        m = 6.64215568e-26 # 40 amu in kg
        omega_rf = self.omega_rf
        joule_to_ev = 6.24150934e18 # conversion factor to take joules -> eV

        E2 = self.compute_squared_field_amplitude(xp, yp, zp)

        return (q**2/(4*m*omega_rf**2))*E2*joule_to_ev # psuedopotential in eV

    def compute_pseudopot_frequencies(self, xp, yp, zp):
        '''
        This is only valid if xp, yp, zp is the trapping position. Return frequency (i.e. 2*pi*omega)
        '''
        d = self.d
        U0 = self.compute_pseudopot(xp, yp, zp)
        ev_to_joule = 1.60217657e-19
        m = 6.64215568e-26 # 40 amu in kg
        d2Udx2 = ev_to_joule*(self.compute_pseudopot(xp + d, yp, zp) - 2*U0 + self.compute_pseudopot(xp - d, yp, zp))/(d**2)
        d2Udy2 = ev_to_joule*(self.compute_pseudopot(xp, yp + d, zp) - 2*U0 + self.compute_pseudopot(xp, yp -d, zp))/(d**2)
        d2Udz2 = ev_to_joule*(self.compute_pseudopot(xp, yp, zp + d) - 2*U0 + self.compute_pseudopot(xp, yp, zp -d ))/(d**2)
        
        '''
        Now d2Udx2 has units of J/m^2. Then w = sqrt(d2Udx2/(mass)) has units of angular frequency
        '''
        
        fx = np.sqrt(abs(d2Udx2)/m)/(2*np.pi)
        fy = np.sqrt(abs(d2Udy2)/m)/(2*np.pi)
        fz = np.sqrt(abs(d2Udz2)/m)/(2*np.pi)
        return [fx, fy, fz]

    def compute_dc_potential_frequencies(self, xp, yp, zp):
        '''
        As always, this is valid only at the trapping position. Return frequency (not angular frequency)
        '''
        joule_to_ev = 6.24150934e18 # conversion factor to take joules -> eV
        m = 6.64215568e-26 # 40 amu in kg
        U0 = self.compute_dc_voltage(xp, yp, zp)
        d2Udx2 = ev_to_joule*(self.compute_dc_voltage(xp + d, yp, zp) - 2*U0 + self.compute_dc_voltage(xp - d, yp, zp))/(d**2)
        d2Udy2 = ev_to_joule*(self.compute_dc_voltage(xp, yp + d, zp) - 2*U0 + self.compute_dc_voltage(xp, yp -d, zp))/(d**2)
        d2Udz2 = ev_to_joule*(self.compute_dc_voltage(xp, yp, zp + d) - 2*U0 + self.compute_dc_voltage(xp, yp, zp -d ))/(d**2)

        fx = np.sqrt(abs(d2Udx2)/m)/(2*np.pi)
        fy = np.sqrt(abs(d2Udy2)/m)/(2*np.pi)
        fz = np.sqrt(abs(d2Udz2)/m)/(2*np.pi)
        return [fx, fy, fz]