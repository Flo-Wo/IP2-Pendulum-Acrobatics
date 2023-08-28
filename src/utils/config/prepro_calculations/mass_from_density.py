import math


def cylinder_mass_from_density(height, radius, density):
    # Calculate the volume of the cylinder
    volume = math.pi * radius**2 * height

    # Calculate the mass using the formula: mass = density * volume
    mass = density * volume

    return mass
