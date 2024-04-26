-- FLOOR
material_props(0.7, 0) -- mirror
shade_albedo(1, 1, 1); -- white
primitive_plane(0, -2, 0, 0, 1, 0)

-- MIRROR BOX
material_props(0.1, 0) -- mirror
shade_albedo(1, 1, 1); -- white
primitive_box(0, 0, 0, 1, 0.5, 1)

-- MISC
material_props(1, 0) -- plastic
shade_albedo(1, 0, 0); -- red
sphere1 = primitive_sphere(-2, 0, 0, 0.5)

material_props(0.2, 0) -- shiny
shade_albedo(0, 1, 0); -- green
sphere2 = primitive_sphere(2, 0, 0, 0.5)

material_props(0.2, 0) -- shiny
shade_emissive(1.7, 1.5, 1.7); -- light
sphere3 = primitive_sphere(0, 0, 2, 0.6)

phase = 0
function update(dt)
    set_position(sphere1, -3 + math.sin(phase), 0, 0)
    set_position(sphere2, 3 + math.sin(phase), 0, 0)
    set_position(sphere3, 0, 0, 3 + math.sin(phase))
    phase = phase + dt * 6
end