""" Tool for visualize the ML model inference, staging process animation
"""
from ursina import *
from csv_parser_pd_ver2 import set_gen_fr_csv_pd_ver3

# create a window
app = Ursina(development_mode=True, borderless=False)

# debug camera
cam = EditorCamera(rotation=(-90, 0, 0))

# light
L1 = PointLight(parent=cam, x=0, y=0, z=-16, color=color.rgba(80, 80, 80))

mesh_paths = [
    [f"../stl/123957/ToothSurface_{q}{n}.stl" for n in range(1, 9)] for q in range(1, 5)]
# Entityes
jaw_up = Entity(scale=(.1, .1, -.1), flipped_faces=True, collider="box") # collider???
jaw_dw = Entity(scale=(.1, .1, -.1), flipped_faces=True, collider="box")

teeth1 = [[Entity(parent=jaw_up, model=tooth) for tooth in q]
          for q in mesh_paths[:2]]
teeth2 = [[Entity(parent=jaw_dw, model=tooth) for tooth in q]
          for q in mesh_paths[2:]]

# draw landmarks
csv_path = r"C:\Projects\torchEncoder\csv\diego_landmarks_over_900.csv"
t1_landmarks = set_gen_fr_csv_pd_ver3(csv_path, one_case="123957.oas")[0][0]
t1_landmarks = t1_landmarks.reshape(-1, 3)  # group by 3
# print("len", t1_landmarks)


# draw lendmarks lower jaw first
# base landmark need for control all batch
lm_down_base = Entity(scale=(0.2, 0.2, -.2),)
lm_down = [Entity(parent=lm_down_base, model='sphere', 
                flipped_faces=False,
                color=color.lime,
                scale=(.5, .5, .5),  # affects only zoom
                position=(lm[0]/2, lm[1]/2, lm[2]/2),
                ) for lm in t1_landmarks[:240//3]]
# then upper
lm_up_base = Entity(model='sphere', scale=(0.2, 0.2, -.2),)
lm_up = [Entity(parent=lm_up_base, model='sphere',
                flipped_faces=False,
                color=color.red,
                scale=(.5, .5, .5),  # affects only zoom
                position=(lm[0]/2, lm[1]/2, lm[2]/2),
                ) for lm in t1_landmarks[240//3:]]


def input(key):
    global teeth
    if key == 'escape':
        quit()

    if key == '1':
        jaw_up.visible = False if jaw_up.visible == True else True
        lm_up_base.visible = False if lm_up_base.visible == True else True

    if key == '2':
        jaw_dw.visible = False if jaw_dw.visible == True else True
        lm_down_base.visible = False if lm_down_base.visible == True else True
    # print(f"jaw_dw.position {jaw_dw.position}")
    # print(f"jaw_dw.rotation {jaw_dw.rotation}")


def update():
    if held_keys['shift']:
        jaw_dw.z += (held_keys['s'] - held_keys['w']) * time.dt
        jaw_dw.y += (held_keys['a'] - held_keys['d']) * time.dt
        jaw_dw.x += (held_keys['z'] - held_keys['x']) * time.dt
    else:
        jaw_dw.rotation_z += (held_keys['s'] - held_keys['w']) * time.dt
        jaw_dw.rotation_y += (held_keys['a'] - held_keys['d']) * time.dt
        jaw_dw.rotation_x += (held_keys['z'] - held_keys['x']) * time.dt


# start running
app.run()
