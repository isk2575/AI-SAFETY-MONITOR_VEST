import cv2 # oepnCV- video processing and drawing
import json # to load zones.json
import time# to calculate PFS
import os # far handling file paths
from pathlib import Path #path handling
import numpy as np
from ultralytics import YOLO
import uuid, datetime
import math









def load_config(zones_json_path):
    with open (zones_json_path, "r") as file:
         config = json.load(file)
    

    frame_w = config.get("frame_width")
    frame_h = config.get("frame_height")
    zones = config.get("zones", [])
    overlay = config.get("overlay", {})
    detector = config.get("detector", {})
    target_class = detector.get("target_classes", ["person"])
    conf = detector.get("conf", 0.20)
    iou = detector.get("iou", 0.5)
    imgsz = detector.get("imgsz", 256)

    if not zones or frame_w is None or frame_h is None:#TESt is zones is not empty,if the frame_w and frame_h have been copied
        print("[ERROR] Invalid or empty zones.json file")
        exit(1)


    print(f"[INFO] Loaded{len(zones)} zone(s) from {zones_json_path}")
    return frame_w, frame_h, zones, overlay, target_class, conf, iou, imgsz
    

video_path = "data/try2.mp4"
zones_path = "config/zones.json"


def open_video(video_path):
    cap= cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"[ERROR] cannot open video:{video_path}")
    
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)


    print(f"[Info] video opened:{vid_w}x{vid_h}@{vid_fps} FPS")
    return cap, vid_w, vid_h, vid_fps 



def draw_zones(frame, zones, overlay_cfg):
    for zone in zones:
      pts = zone["points"]#copy wone points fro, json file
      pts_array = [(int(x), int(y)) for x, y in pts]# turn [string point1, string point2] -> (int(1), int(2))

#convert list of points into numpy array(good for cv2)
      pts_np = np.array(pts_array, dtype = np.int32)
      pts_np = pts_np.reshape((-1, 1, 2))#-1 → tells NumPy to figure out the number of points automatically.#1 → is required because OpenCV wants each point wrapped in its own "array row".#2 → is for the two values in each point: x and y.
      color_hint = zone.get("color_hint", "red")
      if color_hint == "red":
        color = (0,0,255)

      elif color_hint == "green":
        color= (0, 255, 0)

      else:
          color = (0,255,255)


      cv2.polylines(frame, [pts_np], True, color , 2)
      if overlay_cfg.get("show_zone_labels", True):
           label_pos = pts_array[0]
           cv2.putText(frame, zone["id"], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) #this line put the zone name at pts_array[0](top left)
#DANGER_ZONE             SAFE_ZONE
  # ■■■■■■■■■■              ■■■■■■■■■■
  # ■        ■              ■        ■    
  # ■        ■              ■        ■
  # ■■■■■■■■■■              ■■■■■■■■■■    those are example of what the if bloc does










if __name__== "__main__":
    video_path = "data/try2.mp4"
    zones_path = "config/zones.json"
    frame_w, frame_h, zones, overlay, target_class, conf, iou, imgsz =load_config(zones_path)
    # Precompute zone polygons and a severity lookup
    zones_prepared = []
    zone_severity = {}
    for z in zones:
        # (N,2) float32 array for pointPolygonTest
        poly2 = np.array([(int(x), int(y)) for x, y in z["points"]], dtype=np.float32)
        zones_prepared.append((z["id"], poly2))
        zone_severity[z["id"]] = z.get("severity", "low").lower()

    

    vest_model = YOLO("runs/detect/ppe_detector/weights/best.pt")
    person_model = YOLO("yolo11n.pt")

    
    print("[INFO] YOLO11 model loaded:", vest_model)
    print("[INFO] YOLO11 model loaded:", person_model)
    label_map = vest_model.names
    tc = {n.lower() for n in target_class}
    target_cls_ids = [cid for cid, name in label_map.items() if name.lower() in tc]  


    if not target_cls_ids:
        print("[WARN] No target classes matched model labels. Defaulting to 'person'.")
        for cls_id, name in label_map.items():
            if name == "kamizelka":
                target_cls_ids = [cls_id]
                break
        

    print(f"[INFO] Target classes: {target_class}")
    print(f"[INFO] Resolved class IDs: {target_cls_ids}")
        
    cap, vid_w, vid_h, vid_fps = open_video(video_path)
    print("Video size:", vid_w, vid_h)
    print("Zones file size:", frame_w, frame_h)
    fps = 0.0
    prev = time.time()

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/events.jsonl"


    while True:
       ret, frame = cap.read()
       if not ret:
          break
           
          
       h, w = frame.shape[:2]
       if (w != frame_w) or (h != frame_h):
           frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

       t0 = time.time()            # for latency/fps
       zone_hits = set()           # aggregate zone IDs hit this frame
       kept = []                   # detections you'll log/draw


       person_results = person_model.predict(
            source = frame,
            conf = 0.3,
            iou = iou,
            classes = [0],
            imgsz=imgsz,
            verbose = False
        )[0]
       #frame = person_results.plot() 
      

       vest_results = vest_model.predict(
           
           source = frame,
           conf = conf,
           iou = iou,
           classes = target_cls_ids,
           imgsz = imgsz,
           verbose = False
       )[0]
       try:
           frame = vest_results.plot(img = frame)

       except TypeError:
           vplot = vest_results.plot()
       #frame2 = vest_results.plot() 

    
       
       # Process each detection (per-detection logic)
       people=[]
       for box in person_results.boxes:
          """class_id = int(box.cls.item())
          conf_val = float(box.conf.item())
          if class_id not in target_cls_ids:
            continue  # skip unwanted detections"""
        
          # bbox
          x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

          # center point
          cx = (x1 + x2) // 2
          cy = (y1 + y2) // 2
          conf_val = float(box.conf.item())

          people.append({
              "bbox":[x1,y1,x2,y2],
              "center":[cx, cy],
              "confidence":conf_val,
              "has_vest":False
          })

       vests=[]
       for box in vest_results.boxes:
          x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
          cx = (x1+x2)//2
          cy = (y1+y2)//2

          vests.append({
              "center":[cx, cy],
              "bbox": [x1,y1,x2,y2]
          })


       VEST_DISTANCE_THRESHOLD = 150

       for person in people:
        px, py = person["center"]
        for vest in vests:
            vx, vy = vest["center"]
            distance = math.sqrt((px - vx)**2 + (py - vy)**2)
            if distance < VEST_DISTANCE_THRESHOLD:
                person["has_vest"] = True
                break

          
         
          # check zones
        
       violations = []


       for person in people:
          cx, cy = person["center"]
          in_zones = []
          for zone_id, poly2 in zones_prepared:
              inside = cv2.pointPolygonTest(poly2, (float(cx), float(cy)), False) >= 0
              if inside:
                  in_zones.append(zone_id)

          if in_zones:
            if in_zones and not person["has_vest"]:
            # NO VEST - VIOLATION!
                zone_hits.update(in_zones)
                violations.append({
                "bbox": person["bbox"],
                "center": person["center"],
                "confidence": person["confidence"],
                "zones": in_zones,
                "has_vest": False
            })

          x1, y1, x2, y2 = person["bbox"]

       
         

             
          
         
       if not violations:
           decision = "allow"
       else:
           # find severities of all zones hit
           severities = []
           for z in zones:
               if z["id"] in zone_hits:
                   sev = z.get("severity", "low").lower()  # default = low
                   severities.append(sev)
           
           # apply rules
           if "high" in severities:
               decision = "block"
           elif "medium" in severities:
               decision = "escalate"
           else:
               decision = "allow"  # low or untagged zones

       # FPS (simple smoothing)
       now = time.time()
       fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev))
       prev = now

       # Draw frame-level overlay (FPS and decision)
       hits_str = ", ".join(sorted(zone_hits)) if zone_hits else "none"
       color = (50, 220, 50) if decision == "allow" else ((0, 0, 255) if decision == "block" else (0, 165, 255))
       cv2.putText(frame, f"FPS: {fps:.1f} | Decision: {decision} | violations: {len(violations)}| People:{len(people)}",
                   (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
       
       # JSONL event log (one event per frame)
       event = {
           "event_id": str(uuid.uuid4()),
           "timestamp": datetime.datetime.utcnow().isoformat(),
           "decision": decision,
           "zone_hits": list(zone_hits),
           "violations": violations,  # People without vests
           "total_people": len(people),
           "people_with_vests": sum(1 for p in people if p["has_vest"]),
           "latency_ms": round((time.time() - t0) * 1000, 2)
       }

       with open(log_path, "a") as f:
           f.write(json.dumps(event) + "\n")

       # Draw zones and show frame
       draw_zones(frame, zones, overlay)
       cv2.imshow("safety Monitor", frame)
       
       if cv2.waitKey(1) & 0xFF == ord('q'):
          break
       
    cap.release()
    cv2.destroyAllWindows()

