from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import shutil
import os
import uuid

from model_inference import analyze_image  # 你自己的 YOLO 分析函式

app = FastAPI()

# 啟用 CORS，允許從前端（如 Next.js）請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 設定檔案路徑
UPLOAD_DIR = "../public/uploads"
OUTPUT_DIR = "../public/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Pydantic 模型：前端 POST /analyze 的資料結構
class ImageRequest(BaseModel):
    id: str
    year: int
    file_path: str

# ✅ 處理分析請求
@app.post("/analyze")
async def analyze(images: List[ImageRequest]):
    results = []
    for img in images:
        try:
            output_path, coverage = analyze_image(img.file_path, OUTPUT_DIR)

            results.append({
                "id": img.id,
                "coverage": round(coverage, 2),
                "processed_url": f"/outputs/{os.path.basename(output_path)}"  # 只回傳相對 public 路徑
            })
        except Exception as e:
            results.append({
                "id": img.id,
                "error": str(e)
            })

    return results

# ✅ 處理圖片上傳
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 儲存在 uploads 資料夾
        file_ext = file.filename.split(".")[-1]
        temp_filename = f"{uuid.uuid4().hex}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, temp_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"file_path": file_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ 靜態檔案掛載：給前端讀取處理後圖片
app.mount("/outputs", StaticFiles(directory="../public/outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="../public/uploads"), name="uploads")