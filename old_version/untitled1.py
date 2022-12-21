# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:06:09 2022

@author: Lab KYLO
"""
import numpy as np
#Si center
a = np.array([2.528,2.528,3.687])

#O
o1 = np.array([2.083,3.792,4.609])

o2 = np.array([3.792,2.973,2.765])

o3 = np.array([2.973,1.264,4.609])

o4 = np.array([1.264,2.083,2.765])

#center Si - O bonds
Sio1 = np.subtract(a,o1)
Sio2 = np.subtract(a,o2)
Sio3 = np.subtract(a,o3)
Sio4 = np.subtract(a,o4)

Sio1 = Sio1 / np.sqrt((Sio1**2).sum(axis=0))
Sio2 = Sio2 / np.sqrt((Sio2**2).sum(axis=0))
Sio3 = Sio3 / np.sqrt((Sio3**2).sum(axis=0))
Sio4 = Sio4 / np.sqrt((Sio4**2).sum(axis=0))

#Si
b = np.array([2.528,5.056,5.530])

c = np.array([5.056,2.528,1.843])

d = np.array([2.528,0,5.53])

e = np.array([0,2.528,1.843])

# to 轉晶向
b1 = np.subtract(b,a)
b2 = np.subtract(c,a)
b7 = np.subtract(d,a) 
b8 = np.subtract(e,a) 
# outside Si - O bonds
b3 = np.subtract(b,o1)
b4 = np.subtract(c,o2)
b5 = np.subtract(d,o3)
b6 = np.subtract(e,o4)

b1 = b1 / np.sqrt((b1**2).sum(axis=0))
b2 = b2 / np.sqrt((b2**2).sum(axis=0))
b7 = b7 / np.sqrt((b7**2).sum(axis=0))
b8 = b8 / np.sqrt((b8**2).sum(axis=0))


b3 = b3 / np.sqrt((b3**2).sum(axis=0))
b4 = b4 / np.sqrt((b4**2).sum(axis=0))
b5 = b5 / np.sqrt((b5**2).sum(axis=0))
b6 = b6 / np.sqrt((b6**2).sum(axis=0))


desire = np.array([0,0,1])

cross = np.cross(b1,desire)
cross = cross / np.sqrt((cross**2).sum(axis=0))

dot = np.dot(b1,desire)
theta = np.arccos(np.dot(b1,desire))

# phi_axis = (TiltProp['TiltDirectionAngle']+90) /180*np.pi
t = theta
# rotation axis [xyz]
u = cross

c1 = 1-np.cos(t)
# rotation matrix [xyz,xyz]
RM = np.array([
    [np.cos(t)+u[0]**2*c1, u[0]*u[1]*c1-u[2]*np.sin(t), u[0]*u[2]*c1+u[1]*np.sin(t)],
    [u[1]*u[0]*c1+u[2]*np.sin(t), np.cos(t)+u[1]**2*c1, u[1]*u[2]*c1-u[0]*np.sin(t)],
    [u[2]*u[0]*c1-u[1]*np.sin(t), u[2]*u[1]*c1+u[0]*np.sin(t), np.cos(t)+u[2]**2*c1]])

m = np.row_stack((b1,b2,b7,b8,b3,b4,b5,b6,Sio1,Sio2,Sio3,Sio4))

after = np.matmul(RM,m.T).T


b12 = after[1]


desire = np.array([0.942816,0,0])
desire = desire / np.sqrt((desire**2).sum(axis=0))



bb = np.array([b12[0],b12[1],0])
bb = bb / np.sqrt((bb**2).sum(axis=0))

cross = np.cross(bb,desire)
cross = cross / np.sqrt((cross**2).sum(axis=0))
dot = np.dot(bb,desire)
theta = np.arccos(np.dot(bb,desire))

# phi_axis = (TiltProp['TiltDirectionAngle']+90) /180*np.pi
t = theta
# rotation axis [xyz]
u = cross

c1 = 1-np.cos(t)
# rotation matrix [xyz,xyz]
RM = np.array([
    [np.cos(t)+u[0]**2*c1, u[0]*u[1]*c1-u[2]*np.sin(t), u[0]*u[2]*c1+u[1]*np.sin(t)],
    [u[1]*u[0]*c1+u[2]*np.sin(t), np.cos(t)+u[1]**2*c1, u[1]*u[2]*c1-u[0]*np.sin(t)],
    [u[2]*u[0]*c1-u[1]*np.sin(t), u[2]*u[1]*c1+u[0]*np.sin(t), np.cos(t)+u[2]**2*c1]])

# m = np.row_stack((b1,b2,b3,b4))

after1 = np.matmul(RM,after.T).T














