from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ood_detector import OODDetector

app = FastAPI(title="Mole Detector API")

app.add_middleware(
    CORSMiddleware,
    #allow_origins=["*", "https://mole.nicolascolli.com.ar"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

model = YOLO("models/yolo/best.pt")
ood_detector = OODDetector(ood_dir="models/ood")

CONF_THRESHOLD = 0.05  # 5%


def boxes_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def merge_boxes(detections):
    """
    Groups detections that share area (overlap/touch) and, for each group,
    returns the smallest bbox that contains all of them — so a single mole
    doesn't get split into multiple overlapping crops.
    """
    n = len(detections)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    boxes = [d["bbox"] for d in detections]

    # Repeat until stable: merging two boxes can create a bigger box that
    # now overlaps a third one it didn't touch before.
    changed = True
    while changed:
        changed = False
        # Recompute current group bboxes to test overlap correctly
        group_bbox = {}
        for i in range(n):
            r = find(i)
            gx = group_bbox.get(r)
            bx = boxes[i]
            if gx is None:
                group_bbox[r] = list(bx)
            else:
                gx[0] = min(gx[0], bx[0])
                gx[1] = min(gx[1], bx[1])
                gx[2] = max(gx[2], bx[2])
                gx[3] = max(gx[3], bx[3])

        roots = list(group_bbox.keys())
        for a in range(len(roots)):
            for b in range(a + 1, len(roots)):
                ra, rb = roots[a], roots[b]
                if find(ra) != find(rb) and boxes_intersect(group_bbox[ra], group_bbox[rb]):
                    union(ra, rb)
                    changed = True

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        group_dets = [detections[i] for i in idxs]
        x1 = min(d["bbox"][0] for d in group_dets)
        y1 = min(d["bbox"][1] for d in group_dets)
        x2 = max(d["bbox"][2] for d in group_dets)
        y2 = max(d["bbox"][3] for d in group_dets)
        best = max(group_dets, key=lambda d: d["confidence"])
        merged.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": best["confidence"],
            "class_id": best["class_id"],
            "class_name": best["class_name"],
            "merged_from": len(group_dets),
        })
    return merged

templates = Jinja2Templates(directory="../frontend")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "No se pudo leer la imagen")

    results = model.predict(image, conf=0.01, imgsz=640, verbose=False)
    #results = model.predict(image, conf=0.01, verbose=False)
    r = results[0]

    raw_detections = []
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        raw_detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": int(box.cls[0]),
            "class_name": r.names[int(box.cls[0])],
        })

    merged_detections = merge_boxes(raw_detections)

    for det in merged_detections:
        x1, y1, x2, y2 = det["bbox"]
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        x2c, y2c = min(image.width, int(x2)), min(image.height, int(y2))
        crop = image.crop((x1c, y1c, x2c, y2c))
        det["ood"] = ood_detector.score(crop)

    return {
        "detections": merged_detections,
        "image_size": {"width": image.width, "height": image.height},
    }