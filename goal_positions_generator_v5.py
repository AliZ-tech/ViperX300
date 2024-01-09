import numpy as np
import pandas as pd
from Kinematics_ViperX_300_6DOF import *
import matplotlib.pyplot as plt

# Box and Gripper length (mm)
box_l = 94
box_w = 61
grip_l = 68
grip_w = 61
grip = 40

joint_id = np.array([1, 23, 45, 6, 7, 8, 9])
q_correction = np.array([0, -7, 10, 7, 7, 18])
grip_open = 2800
grip_close = 2400

rest_point = np.array([2048, 810, 3100, 2045, 1568, 2065])

ref_point0 = np.array([300, 0, 300])

box_point0 = np.array([390, 40, grip_l + box_l - grip])
box_point1 = box_point0 + np.array([0, 0, 200])

scanner_point0 = np.array([-150, -195, 135])    # Scanner tip
gap = 5
scanner_point1 = scanner_point0 + np.array([0, 0, 120]) 
scanner_adjust_s1 = np.array([-10, -25, grip_l + box_l - grip + gap])
scanner_adjust_s2 = np.array([grip_l+box_l-grip-45, -10, box_l/2 + gap])
scanner_adjust_s3 = np.array([grip_l+box_l-grip-45, -27, box_w/2 + gap])

temp_point1 = np.array([-140, 170, 250])
temp_point2 = np.array([-140, 170, 50]) - np.array([box_l/2 + grip_l - grip, 0, 0])
temp_point3 = temp_point2 + np.array([0, 0, 100])
temp_point4 = np.array([-140, 170, 50]) + np.array([box_l/2 + grip_l - grip, 0, 0]) + np.array([0, 0, 100])
temp_point5 = temp_point4 + np.array([0, 0, -100])

pi = np.pi
data = []
step = 1
q0 = np.zeros(6)

def go_to_joint(q, grip, time, scan=0, error=100):
    # recieves joint values, and 
    # adds required commands to data
    # q: join angles (6x1)
    # grip: grip status (open / closed)
    # step: # of steps/moves in the sequence
    # time: length of the move
    global step, data
    data_step = []
    for i in range(6):
        data_step.append(int(np.round(q[i])))
    if grip == 'open':
        data_step.append(grip_open)
    else:
        data_step.append(grip_close)
    data_step.append(time)
    data_step.append(scan)
    data_step.append(error)
    data.append(data_step)
    step += 1

def go_to(pos, rot, grip, time, scan=0, error=100):
    # recieves cartesian position and rot matrix, and 
    # adds required commands to data
    # pos: 3x1 postion vector
    # rot: 3x3 rotation matrix
    # grip: grip status (open / closed)
    # step: # of steps/moves in the sequence
    # time: length of the move
    global step, data, q0
    q = inv_kin(pos, rot, q0)
    q0 = q.copy()
    pos0, rot0 = for_kin(q)
    print(step, np.linalg.norm(pos0 - pos))
    q = q / (2 * pi) * 4096 + 2048
    q = q + q_correction 
    go_to_joint(q, grip, time, scan, error)

def scan_s1():
    # hands on the process from scanner_point1, covers the lower 4x2 full surface, and hands it back scanner_point1
    go_to(scanner_point0 + scanner_adjust_s1, rot_y(pi/2), 'close', 1500)
    for i in range(4):
        for j in range(2):
            go_to(scanner_point0 + scanner_adjust_s1 + i * np.array([1, 0, 0]) * 17.78 + j * np.array([0, 1, 0]) * 17.78, rot_y(pi/2), 'close', 250, 1, 10)
    go_to(scanner_point1 + scanner_adjust_s1, rot_y(pi/2), 'close', 1500)

def scan_s2(rot):
    # hands on the process from scanner_point1, covers the lower 4x2 half surface, and hands it back scanner_point1
    go_to(scanner_point0 + scanner_adjust_s2, rot, 'close', 1500)
    for i in range(2):
        for j in range(2):
            go_to(scanner_point0 + scanner_adjust_s2 + i * np.array([1, 0, 0]) * 17.78 + j * np.array([0, 1, 0]) * 17.78, rot, 'close', 250, 1, 10)
    go_to(scanner_point1 + scanner_adjust_s2, rot, 'close', 1500)

def scan_s3(rot):
    # hands on the process from scanner_point1, covers the lower 4x4 half surface, and hands it back scanner_point1
    go_to(scanner_point0 + scanner_adjust_s3, rot, 'close', 1500)
    for i in range(2):
        for j in range(4):
            go_to(scanner_point0 + scanner_adjust_s3 + i * np.array([1, 0, 0]) * 17.78 + j * np.array([0, 1, 0]) * 17.78, rot, 'close', 250, 1, 10)
    go_to(scanner_point1 + scanner_adjust_s3, rot, 'close', 1500)

def scan_half():
    scan_s1()
    go_to(scanner_point1 + scanner_adjust_s2, rot_y(pi), 'close', 1500)
    scan_s2(rot_y(pi))
    go_to(scanner_point1 + scanner_adjust_s3, np.matmul(rot_x(-pi/2), rot_y(pi)), 'close', 1500)
    scan_s3(np.matmul(rot_x(-pi/2), rot_y(pi)))
    go_to(scanner_point1 + scanner_adjust_s3, np.matmul(rot_x(pi/2), rot_y(pi)), 'close', 1500)
    scan_s3(np.matmul(rot_x(pi/2), rot_y(pi)))
    go_to(scanner_point1 + scanner_adjust_s2, np.matmul(rot_x(pi), rot_y(pi)), 'close', 1500)
    scan_s2(np.matmul(rot_x(pi), rot_y(pi)))

# start from rest and fetch a box
go_to(ref_point0, np.eye(3), 'open', 1500)
go_to(box_point1, rot_y(pi/2), 'open', 2000)
go_to(box_point0, rot_y(pi/2), 'open', 1500, error=10)
go_to(box_point0, rot_y(pi/2), 'close', 1500)
go_to(box_point1, rot_y(pi/2), 'close', 1500)
q0 = np.zeros(6)
go_to(scanner_point1 + scanner_adjust_s1, rot_y(pi/2), 'close', 3000)

# scan first half
scan_half()

# rotate the box
go_to(ref_point0, rot_y(pi/2), 'close', 1500)
go_to(temp_point1, rot_y(pi/2), 'close', 3000)
go_to(temp_point2, rot_x(pi), 'close', 1500, error=10)
go_to(temp_point2, rot_x(pi), 'open', 1500)
go_to(temp_point3, np.matmul(rot_y(pi/4), rot_x(pi)), 'open', 1500)
# q0 = np.zeros(6)
go_to(temp_point1, rot_y(pi/2), 'open', 1500)
# q0 = np.zeros(6)
go_to(temp_point4, rot_y(pi), 'open', 1500)
go_to(temp_point5, rot_y(pi), 'open', 1500, error=10)
go_to(temp_point5, rot_y(pi), 'close', 1500)
go_to(temp_point1, rot_y(pi), 'close', 1500)
q0 = np.zeros(6)
go_to(ref_point0, rot_y(pi/2), 'close', 1500)
q0 = np.zeros(6)
go_to(scanner_point1 + scanner_adjust_s1, rot_y(pi/2), 'close', 3000)

# scan second half
scan_half()

# return box
go_to(box_point1, rot_y(pi/2), 'close', 2500)
go_to(box_point0, rot_y(pi/2), 'close', 1500)
go_to(box_point0, rot_y(pi/2), 'open', 1500)
go_to(box_point1, rot_y(pi/2), 'open', 1500)
go_to(ref_point0, np.eye(3), 'open', 1500)

# rest position
go_to_joint(rest_point, 'open', 1500)

# Export the data to a csv file
df = pd.DataFrame(data, columns = joint_id.tolist() + ['move_time', 'scan', 'error_thereshold'])
print(df)
df.to_csv('C:\\DynamixelSDK-3.7.31\\python\\tests\\protocol2_0\\goal_position_v6.csv', index=False)
# print(df[joint_id[1]].iloc[110])

# drawing the sequence of commands
join_limits = np.array([[0, 1189, 858, 0, 594, 0, 3045], [4095, 3396, 3242, 4095, 3293, 4095, 4095]])
p = len(data)
data_j = np.array(data)[:, 0:7]

fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(15, 10))
for i in range(7):
    # axs[i].plot(df['Goal_pos'].iloc[np.arange(i, len(data), 7)].reset_index(drop=True))
    axs[i].plot(data_j[:, i])
    axs[i].plot(np.ones(p) * join_limits[0, i], '--')
    axs[i].plot(np.ones(p) * join_limits[1, i], '--')
    axs[i].set_ylabel('Join ' + str(joint_id[i]))
    axs[i].set_ylim([0, 4028])
    axs[i].grid()
plt.show()