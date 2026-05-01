#!/usr/bin/env python3
"""FastAPI service for offline video verification uploads.

This service exposes a small upload endpoint that runs the existing
`tools/process_video.py` pipeline and returns a downloadable annotated video.
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tools.process_video import process_video


ROOT = Path(__file__).resolve().parents[1]
VERIFY_DIR = ROOT / "data" / "offline_verify"
INPUT_DIR = VERIFY_DIR / "inputs"
OUTPUT_DIR = VERIFY_DIR / "outputs"
SUMMARY_DIR = VERIFY_DIR / "summaries"

for directory in (INPUT_DIR, OUTPUT_DIR, SUMMARY_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="MoCap Offline Verification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/verify/files", StaticFiles(directory=str(OUTPUT_DIR)), name="verify_files")


@app.get("/verify/health")
def verify_health() -> dict:
    return {"status": "ok", "service": "offline-verify"}


@app.post("/verify/upload")
def verify_upload(
    file: UploadFile = File(...),
    max_frames: int = Form(0),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use .mp4, .mov, .avi, .mkv, or .m4v",
        )

    upload_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    safe_stem = Path(file.filename).stem.replace(" ", "_")

    input_path = INPUT_DIR / f"{upload_id}_{safe_stem}{suffix}"
    output_path = OUTPUT_DIR / f"{upload_id}_{safe_stem}_annotated.mp4"
    summary_path = SUMMARY_DIR / f"{upload_id}_{safe_stem}_summary.json"

    try:
        with input_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        stats = process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            max_frames=max(0, int(max_frames)),
        )

        summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        return {
            "status": "success",
            "upload_id": upload_id,
            "input_name": file.filename,
            "annotated_video": output_path.name,
            "annotated_video_url": f"/verify/files/{output_path.name}",
            "summary": stats,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc


@app.get("/verify/download/{filename}")
def verify_download(filename: str):
    target = OUTPUT_DIR / os.path.basename(filename)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(target), filename=target.name, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tools.offline_verify_api:app", host="0.0.0.0", port=8000, reload=False)
