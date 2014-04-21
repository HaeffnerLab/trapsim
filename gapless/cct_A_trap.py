'''
Example of the gapless approximation code. Refers to the CCT A trap.
'''

from gapless import World

'''
Add all the electrodes electrodes
'''
gap = 5.

smallrfx1 = -50.
smallrfx2 = -185.
bigrfx1 = 50.
bigrfx2 = 320.
cx1 = -45.
cx2 = 45.

rfy1 = -5280.
rfy2 = 9475.

lx1 = -490.
lx2 = -190.
rx1 = 325.
rx2 = 625.

# y coords of the dc electrodes
y_ranges = [(0, 300.),
            (305., 605.),
            (610., 910.),
            (915., 1215.),
            (1220., 1320.),
            (1325., 1625.),
            (1630., 1930.),
            (1935., 2235),
            (2240., 2540.),
            (2545., 2995.),
            (3000., 3450.)
            ]

''' Now build your own world '''
w = World()
# first build the left electrodes
for num, yr in zip( range(1, 12), y_ranges):
    w.add_electrode(str(num), (lx1, lx2), yr, 'dc')
# now the right electrodes
for num, yr in zip( range(12, 23), y_ranges):
    w.add_electrode(str(num), (rx1, rx2), yr, 'dc')
# add the center
w.add_electrode('23', (cx1, cx2), (rfy1, rfy2), 'dc' )
# add the RF
w.add_electrode('rf1', (smallrfx1, smallrfx2), (rfy1, rfy2), 'rf')
w.add_electrode('rf2', (bigrfx1, bigrfx2), (rfy1, rfy2), 'rf')