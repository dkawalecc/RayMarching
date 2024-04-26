material_props(0.1, 0) -- shiny
shade_emissive(1.7, 1.7, 1.7); -- light
primitive_sphere(0, 0, -1.5, 0.25)
primitive_sphere(0, 0, 0.5, 0.25)

-- MIRROR BOX
material_props(0.1, 0) -- mirror
shade_albedo(1, 1, 1); -- white
primitive_box(0, 0, 3, 5.5, 3, 0.5)

primitive_box(5, 0, 0, 0.5, 3, 3)
primitive_box(-5, 0, 0, 0.5, 3, 3)

AMOUNT_OF_BOXES = 4
boxes = {}

for i = 0, AMOUNT_OF_BOXES-1, 1
do
    boxes[i] = primitive_box(0, 0, 1-i, 1, 1, 0.1) 
end


phase = 0
function update(dt)
   for i =  0, AMOUNT_OF_BOXES-1, 1
      do  
          set_position(boxes[i], 2*math.sin(phase+i*2*math.pi/AMOUNT_OF_BOXES), 0, 1-i)
      end
      phase = phase + dt * 2
end