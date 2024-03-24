#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:34:34 2023

@author: asier.urio
"""
import pandas as pd
datasets=[
    'a_1C2D1kLinear',
    'b_4C2D800Linear',
    'c_4C2D3200Linear',
    'd_3C2D2400Spiral',
    # 'e_4C3D20kLinear',
    # 'f_5C5D1kLinear',
    # 'g_2C3D4kHelix',
    'h_2C2D200kHelix',
    'i_4C2D4kStatic',
    ]


url='https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_'
p='.csv?ref_type=heads&inline=false'
for f in datasets:
    ds=pd.read_csv(url+f+p)
    ds.columns=['x','y','c']
    ds.plot.scatter(x='x',y='y')
    
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d')
plt.scatter(ds['x'],ds['y'],ds['z'])