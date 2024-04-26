-- FLOOR
material_props(1, 0) -- rough
-- shade_albedo(0, 0, 0); -- black
shade_albedo(0.3, 0.3, 0.3); -- white
primitive_plane(0, -1.1, 0, 0, 1, 0)

shade_albedo(0.6, 0.9, 0.5); -- white

-- Spheres
AMOUNT_OF_SPHERES = 5
for i = 0, AMOUNT_OF_SPHERES-1, 1
do
    radius = 1
    padding = 0.2

    material_props((i/(AMOUNT_OF_SPHERES-1)) ^ 2, 0) -- mirror
    primitive_sphere((i - (AMOUNT_OF_SPHERES-1) / 2) * (radius*2 + padding), 0, 0, radius)
end
