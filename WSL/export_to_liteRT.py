# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:46:30 2026

@author: GKoinig
"""

from ultralytics import YOLO
model = YOLO("flo_model.pt")
model.export(
    format="litert",
    imgsz=1024,
    nms=None,
)