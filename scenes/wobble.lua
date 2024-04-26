wobble(0.05, 10, 1);

-- FLOOR
material_props(0.7, 0) -- mirror
shade_albedo(1, 1, 1); -- white
primitive_plane(0, -1, 0, 0, 1, 0)

-- MIRROR BOX
material_props(0.1, 0) -- mirror
shade_albedo(1, 1, 1); -- white
primitive_cuboid(0, 0, 0, 1, 0.5, 1)

-- MISC
material_props(1, 0) -- plastic
shade_albedo(1, 0, 0); -- red
primitive_sphere(-2, 0, 0, 0.5)

material_props(0.2, 0) -- shiny
shade_albedo(0, 1, 0); -- green
primitive_sphere(2, 0, 0, 0.5)

material_props(0.2, 0) -- shiny
shade_albedo_and_emissive(0, 0, 0, 1.7, 1.5, 1.7); -- light
primitive_sphere(0, 0, 2, 0.6)
