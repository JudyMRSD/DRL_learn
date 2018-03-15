from collections import namedtuple

a = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
b = a(0, 1, 2, 3, 4)

print(b.action)
