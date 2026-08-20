"""Find a sensor geometry whose simulated fan resembles the real tank captures."""
import numpy as np, itertools
from varuna.validation import tank_scene
from varuna.acoustics import ForwardLookingSonar, preset
from ultralytics import YOLO
import os
model = YOLO(os.path.expanduser('~/dev/rakshatech/ml/models/fls_yolov8s.pt'))

objs = [("tire", 2.6, -0.35, 0.4), ("bottle", 3.6, 0.5, 1.1),
        ("can", 1.9, 0.3, 0.0), ("valve", 4.4, -0.6, 0.7)]
print(f"{'alt':>5}{'pitch':>7}{'rmax':>6}  {'fill%':>6}{'dets':>6}  best")
best=[]
for alt, pitch, rmax in itertools.product((0.35,0.6,1.0,1.6), (4,7,11,16), (4.0,6.0,9.0)):
    scene, truth = tank_scene(objs, floor_z=-3.0, extent=30.0)
    cfg = preset('aris', seed=5, r_min=0.5, r_max=rmax, ssc_g_per_l=0.3)
    f = ForwardLookingSonar(cfg, scene)
    fr = f.ping([0,0,-3.0+alt,0,np.radians(pitch),0])
    img = fr.to_cartesian(560)
    fill = float((img>0.06).mean())
    u8=(np.clip(img,0,1)*255).astype(np.uint8)
    rgb=np.stack([u8]*3,-1)
    r=model.predict(rgb, conf=0.20, imgsz=640, verbose=False)[0]
    nd = 0 if r.boxes is None else len(r.boxes)
    top = ''
    if nd: 
        i=int(np.argmax(r.boxes.conf.cpu().numpy()))
        top=f"{model.names[int(r.boxes.cls[i])]} {float(r.boxes.conf[i]):.2f}"
    print(f"{alt:5.2f}{pitch:7d}{rmax:6.1f}  {100*fill:6.1f}{nd:6d}  {top}")
    best.append((nd, fill, alt, pitch, rmax))
best.sort(reverse=True)
print()
print("top:", best[:3])
