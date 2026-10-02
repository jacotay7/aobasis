"""The shortest useful aobasis program: KL modes for a DM, saved to disk.

    python examples/quickstart.py
"""

import aobasis

positions = aobasis.make_circular_actuator_grid(telescope_diameter=10.0, grid_size=20)  # (N, 2), metres
kl = aobasis.KLBasisGenerator(positions, fried_parameter=0.16, outer_scale=30.0)
m2c = kl.generate(n_modes=50, ignore_piston=True)  # (n_actuators, n_modes)

print(f"M2C: {m2c.shape[0]} actuators x {m2c.shape[1]} modes")
print(kl.report())
kl.save("kl_m2c.npz")
print("saved kl_m2c.npz")
