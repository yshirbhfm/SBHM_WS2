# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:06:09 2022

@author: Lab KYLO
"""
import numpy as np

a = np.array([3.42,6.43,-1.723])
#O
o1 = np.array([2.148,7.188,-2.35])

o2 = np.array([2.942,4.97,-1.175])

o3 = np.array([4.588,6.255,-2.845])

o4 = np.array([4.003,7.323,-0.495])


o1 = np.subtract(a,o1)
o2 = np.subtract(a,o2)
o3 = np.subtract(a,o3)
o4 = np.subtract(a,o4)

o1 = o1 / np.sqrt((o1**2).sum(axis=0))
o2 = o2 / np.sqrt((o2**2).sum(axis=0))
o3 = o3 / np.sqrt((o3**2).sum(axis=0))
o4 = o4 / np.sqrt((o4**2).sum(axis=0))

#Si
b = np.array([5.17,7.498,0.627])

c = np.array([0.875,7.946,-2.977])

d = np.array([5.17,5.362,-4.073])

e = np.array([3.42,3.51,-0.627])

b1 = np.subtract(a,b)
b2 = np.subtract(a,c)
b3 = np.subtract(a,d)
b4 = np.subtract(a,e)

b1 = b1 / np.sqrt((b1**2).sum(axis=0))
b2 = b2 / np.sqrt((b2**2).sum(axis=0))
b3 = b3 / np.sqrt((b3**2).sum(axis=0))
b4 = b4 / np.sqrt((b4**2).sum(axis=0))


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

m = np.row_stack((b1,b2,b3,b4,o1,o2,o3,o4))

after = np.matmul(RM,m.T).T


b2 = after[1]

desire = np.array([0.942816,0,0])
desire = desire / np.sqrt((desire**2).sum(axis=0))

bb = np.array([b2[0],b2[1],0])
bb = bb / np.sqrt((bb**2).sum(axis=0))

cross = np.cross(bb,desire)
cross = cross / np.sqrt((cross**2).sum(axis=0))
dot = np.dot(bb,desire)
theta = np.arccos(np.dot(bb,desire))

# phi_axis = (TiltProp['TiltDirectionAngle']+90) /180*np.pi
t = theta
# rotation axis [xyz]
u = [0,0,1]

c1 = 1-np.cos(t)
# rotation matrix [xyz,xyz]
RM = np.array([
    [np.cos(t)+u[0]**2*c1, u[0]*u[1]*c1-u[2]*np.sin(t), u[0]*u[2]*c1+u[1]*np.sin(t)],
    [u[1]*u[0]*c1+u[2]*np.sin(t), np.cos(t)+u[1]**2*c1, u[1]*u[2]*c1-u[0]*np.sin(t)],
    [u[2]*u[0]*c1-u[1]*np.sin(t), u[2]*u[1]*c1+u[0]*np.sin(t), np.cos(t)+u[2]**2*c1]])

# m = np.row_stack((b1,b2,b3,b4))

after1 = np.matmul(RM,after.T).T













