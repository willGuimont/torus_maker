import fractions

import numpy as np
from stl import mesh

resolution_curve = 1_500
resolution_inner = 10
radius_a = 42
radius_b = 15
r = radius_a / radius_b
radius_tube = 2.5

num_turns = fractions.Fraction(radius_a, radius_b)

# Create vertices
vertices = []
for t in np.linspace(0, num_turns.denominator * 2 * np.pi, num=resolution_curve):
    angle_a = np.fmod(t, 2 * np.pi)
    angle_b = np.fmod(t * r, 2 * np.pi)

    x = (radius_a + radius_b * np.cos(angle_b)) * np.cos(angle_a)
    y = (radius_a + radius_b * np.cos(angle_b)) * np.sin(angle_a)
    z = radius_b * np.sin(angle_b)
    pos = np.array([x, y, z])

    grad = np.array([np.sin(t) * (-(radius_a + radius_b * np.cos(r * t))) - radius_b * r * np.cos(t) * np.sin(r * t),
                     np.cos(t) * (radius_a + radius_b * np.cos(r * t)) - radius_b * r * np.sin(t) * np.sin(r * t),
                     radius_b * r * np.cos(t * r)])
    tangent = grad / np.linalg.norm(grad)

    normal_1 = np.cross(pos / np.linalg.norm(pos), tangent)
    normal_1 = normal_1 / np.linalg.norm(normal_1)

    normal_2 = np.cross(tangent, normal_1)
    normal_2 = normal_2 / np.linalg.norm(normal_2)

    for inner_angle in np.linspace(0, 2 * np.pi, num=resolution_inner):
        dx = normal_1 * np.cos(inner_angle) * radius_tube
        dy = normal_2 * np.sin(inner_angle) * radius_tube
        p_tube = pos + dx + dy
        vertices.append(p_tube)

# Create faces
faces = []
for i in range(resolution_inner):
    for j in range(resolution_curve):
        a = (i, j)
        b = (i + 1, j)
        c = (i, j + 1)
        d = (i, j - 1)
        e = (i - 1, j)
        # First triangle
        f_t = []
        for p in [a, b, c]:
            x, y = p[0] % resolution_inner, p[1] % resolution_curve
            f_t.append(y * resolution_inner + x)
        faces.append(f_t)
        # Second triangle
        s_t = []
        for p in [a, d, e]:
            x, y = p[0] % resolution_inner, p[1] % resolution_curve
            s_t.append(y * resolution_inner + x)
        faces.append(s_t)

# Create the mesh
vertices = np.array(vertices)
faces = np.array(faces)
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j], :]

# Write the mesh to file "cube.stl"
cube.save('models/curve.stl')
