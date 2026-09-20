import inspect, re
from gymnasium.envs.box2d import car_racing as cr
for i, l in enumerate(inspect.getsource(cr).splitlines(), 1):
    if re.search(r"reward|terminated|truncated|lap_complete|off_track|is_touching", l):
        print(f"{i:5d} {l.rstrip()}")
