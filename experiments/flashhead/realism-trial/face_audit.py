"""Offline diagnostic only: landmark estimates are not a realism score."""
import json
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

path = '/experiment/minute/pro-eager.mp4'
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
rows = []
def ear(points, indices):
    p = points[indices]
    return float((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*np.linalg.norm(p[0]-p[3])))
with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as model:
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        result = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            p = np.array([(v.x*frame.shape[1],v.y*frame.shape[0]) for v in result.multi_face_landmarks[0].landmark])
            rows.append({'time':i/fps,'left_eye':ear(p,[33,160,158,133,153,144]),'right_eye':ear(p,[362,385,387,263,373,380]),'nose':p[1].tolist(),'mouth_open':float(np.linalg.norm(p[13]-p[14]))})
        i += 1
cap.release()
openness = np.array([(r['left_eye']+r['right_eye'])/2 for r in rows])
baseline = float(np.percentile(openness,80))
events=[]; start=None
for row, val in zip(rows,openness):
    low = val < baseline * 0.65
    if low and start is None: start = row['time']
    elif not low and start is not None:
        events.append({'start':start,'end':row['time'],'seconds':row['time']-start});start=None
if start is not None: events.append({'start':start,'end':rows[-1]['time']+1/fps,'seconds':rows[-1]['time']+1/fps-start})
report={'fps':fps,'frames':i,'detected_frames':len(rows),'method':'MediaPipe relative eye aperture; exploratory geometry, not a validated realism or blink score','eye_open_baseline':baseline,'eye_aperture_below_65pct_events':events,'rows':rows}
Path('/experiment/quality-v3').mkdir(exist_ok=True)
Path('/experiment/quality-v3/face-audit.json').write_text(json.dumps(report))
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
