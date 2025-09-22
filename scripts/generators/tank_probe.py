import os
import numpy as np

TRAVEL_SPEED=5000
PROBE_SPEED=250
START_Z = -100     # xy probing depth = -72.5 z swtich to spoilboard + 10 - 16 mm probing point min tank thickness - z probe from xy (21mm)
OFFSET_XY_MM = 10   # distance to retract from tank after probing xy
OFFSET_Z_MM = 60    # 21mm from z to xy switch + 18mm clearnce + 21 offset measuremtn pt
OFFSET_XY_SWITCH_MM = 37.4 # distance from xy switch to z probe position = 11.4 (switch z delta) + 16 (grove inset dist) + OFFSET_X_MM (10)

def shell_probe(type):
    inner_coords = np.loadtxt(f"{type}concentric_inner.csv", delimiter=',', skiprows=1)     # Load inner coordinates (X,Y)
    outer_coords = np.loadtxt(f"{type}concentric_outer.csv", delimiter=',', skiprows=1)     # Load outer coordinates (X,Y)

    with open(os.path.expanduser(f'./{type}probe.gcode'), 'w') as output:
        for line in startup:
            output.write(line + '\n')    # Write startup G-code lines

        for index in range(len(inner_coords)):
            if index == 0:
                output.write(f"G0 X{inner_coords[index][0]:.4f} Y{inner_coords[index][1]:.4f} F{TRAVEL_SPEED}\n")   # Move to inner coordinate (X,Y)
            output.write(f"G0 X{inner_coords[index][0]:.4f} Y{inner_coords[index][1]:.4f} Z{START_Z} F{TRAVEL_SPEED}\n")   # Move to inner coordinate (X,Y)
            
            if index == 0:
                output.write(f"G0 Z{START_Z} F{TRAVEL_SPEED / 4}\n")    # Lower Z to starting height (first point only)
            
            output.write(f"G38.2 X{outer_coords[index][0]:.4f} Y{outer_coords[index][1]:.4f} F{PROBE_SPEED}\n") # move outer coordinate (X,Y,Z)

            angle = np.arctan2(outer_coords[index][1] - inner_coords[index][1], outer_coords[index][0] - inner_coords[index][0])    # Calculate angle between inner and outer points
            x_offset = (-OFFSET_XY_MM) * np.cos(angle)  # X offset backward from probe line
            y_offset = (-OFFSET_XY_MM) * np.sin(angle)  # Y offset backward from probe line
            x_offset_2 = (OFFSET_XY_SWITCH_MM) * np.cos(angle)  # X offset backward from probe line
            y_offset_2 = (OFFSET_XY_SWITCH_MM) * np.sin(angle)  # Y offset backward from probe line

            output.write("G91\n")                                                               # Switch to relative positioning                                                         
            output.write(f"G0 X{x_offset:.4f} Y{y_offset:.4f} F{TRAVEL_SPEED / 4}\n")           # Move backward from outer point for clearnce
            output.write(f"G0 Z{OFFSET_Z_MM} F{TRAVEL_SPEED / 4}\n")                            # Lift probe
            output.write(f"G0 X{x_offset_2:.4f} Y{y_offset_2:.4f} F{TRAVEL_SPEED / 4}\n")       # Move forward to z probe points
            output.write("G90\n")                                                               # Switch back to absolute positioning
            output.write(f"G38.2 Z-80 F{PROBE_SPEED}\n")                                        # Probe down along Z until contact (up to -80mm)
            output.write(f"G0 Z{START_Z + OFFSET_Z_MM} F{TRAVEL_SPEED / 4}\n")                  # Retract probe back up to start Z
            output.write("G91\n")                                                               # Switch back to absolute positioning
            output.write(f"G0 X{-x_offset_2:.4f} Y{-y_offset_2:.4f} F{TRAVEL_SPEED / 4}\n")     # Switch to relative positioning
            output.write("G90\n")                                                               # Switch back to absolute positioning


# Example G-Code commands
startup = [
    #"$X",           # Unlock alarm state (if present)
    #"$H",           # Execute homing cycle
    #"G21",          # Set units to mm
    #"G90",          # Absolute positioning
]

shell_probe('a')
shell_probe('b')


