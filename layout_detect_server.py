"""Standalone layout detection API using PaddleX PP-DocLayoutV3.

Provides /layout-detect endpoint that returns detected regions with coordinates.
Runs on port 8081 alongside the PP-OCRv6 service on port 8080.
"""
import base64
import io
import logging
import tempfile
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from paddlex import create_pipeline

logger = logging.getLogger("layout-detect")
logging.basicConfig(level=logging.INFO)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info("Loading PP-DocLayoutV3 layout detection pipeline (CPU mode)...")
        _pipeline = create_pipeline(pipeline="layout_parsing", device="gpu:0")
        logger.info("Layout pipeline loaded")
    return _pipeline


@asynccontextmanager
async def lifespan(app):
    get_pipeline()
    yield


app = FastAPI(title="Layout Detection API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/layout-detect")
async def layout_detect(request: dict):
    """Detect layout regions in an image.

    Input: {"image": "<base64-encoded image>"}
    Output: {"regions": [{"label": "...", "bbox": [x1,y1,x2,y2], "score": 0.95}, ...]}
    """
    image_b64 = request.get("image", "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' field")

    try:
        # Decode image and save to temp file (PaddleX needs file path, not PIL Image)
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)

        # Write to temp file — PaddleX only accepts str or numpy array
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        pipeline = get_pipeline()
        result = pipeline.predict(tmp_path)

        regions = []
        for res in result:
            # Result is a LayoutParsingResult object, not a dict
            if hasattr(res, 'layout_det_res'):
                ldr = res.layout_det_res
                boxes = ldr.boxes if hasattr(ldr, 'boxes') else []
            elif isinstance(res, dict):
                ldr = res.get("layout_det_res", {})
                boxes = ldr.get("boxes", []) if isinstance(ldr, dict) else []
            else:
                boxes = []

            for box in boxes:
                if not isinstance(box, dict):
                    continue
                label = box.get("label", "text")
                coordinate = box.get("coordinate")
                score = box.get("score", 0)
                if coordinate and len(coordinate) >= 4:
                    regions.append({
                        "label": label,
                        "bbox": [int(float(c)) for c in coordinate[:4]],
                        "score": round(float(score), 3),
                    })

        # Sort top-to-bottom reading order
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return {"regions": regions}

    except Exception as e:
        logger.exception("Layout detection failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
