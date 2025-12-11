# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io
from daltonlens import simulate
import cv2  # <<< تمت إضافة مكتبة OpenCV لمعالجة الفيديو
import tempfile  # <<< تمت إضافة مكتبة الملفات المؤقتة

# main_unified.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io
import cv2
import tempfile

app = FastAPI(
    title="Unified Color Correction API",
    description="واجهة برمجية ذكية لتصحيح الصور والفيديوهات لمرضى عمى الألوان عبر نقطة نهاية واحدة."
)

# ربط الأرقام بأنواع عمى الألوان
cvd_types = {
    1: simulate.Deficiency.PROTAN,
    2: simulate.Deficiency.DEUTAN,
    3: simulate.Deficiency.TRITAN,
}

simulator = simulate.Simulator_Machado2009()


def daltonize_with_severity(original, simulated, severity=1.0):
    error = original.astype(np.float32) - simulated.astype(np.float32)
    corrected = original.astype(np.float32) + (error * severity)
    corrected = np.clip(corrected, 0, 255).astype("uint8")
    return corrected


# -------------------------------------------------------------------
# │                      الوظائف المساعدة للمعالجة                    │
# -------------------------------------------------------------------

async def handle_image_correction(file: UploadFile, cvd_type_enum, severity: float):
    """يعالج ملف الصورة ويعيد استجابة البث."""
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.asarray(img)
    except Exception:
        raise HTTPException(status_code=400, detail="ملف الصورة غير صالح.")

    simulated = simulator.simulate_cvd(img_np, cvd_type_enum, severity=severity)
    corrected = daltonize_with_severity(img_np, simulated, severity)
    corrected_img = Image.fromarray(corrected)

    buf = io.BytesIO()
    corrected_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


async def handle_video_correction(file: UploadFile, cvd_type_enum, severity: float):
    """يعالج ملف الفيديو ويعيد استجابة البث."""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video_file:
        content = await file.read()
        temp_video_file.write(content)
        temp_video_file.flush()

        cap = cv2.VideoCapture(temp_video_file.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="ملف الفيديو تالف أو غير مدعوم.")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
            output_path = temp_output_file.name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # إعادة فتح الملف للقراءة بواسطة process_video_frames
            cap_process = cv2.VideoCapture(temp_video_file.name)
            try:
                while True:
                    ret, frame = cap_process.read()
                    if not ret: break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    simulated = simulator.simulate_cvd(frame_rgb, cvd_type_enum, severity=severity)
                    corrected_rgb = daltonize_with_severity(frame_rgb, simulated, severity)
                    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
                    out.write(corrected_bgr)
            finally:
                cap_process.release()
                out.release()

            with open(output_path, "rb") as f:
                video_bytes = f.read()

    return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4", headers={
        "Content-Disposition": f"attachment; filename=corrected_video.mp4"
    })


# -------------------------------------------------------------------
# │                <<< نقطة النهاية الموحدة والذكية >>>              │
# -------------------------------------------------------------------
@app.post("/correct", tags=["Unified Correction"])
async def correct_media(
        file: UploadFile = File(..., description="ملف الصورة أو الفيديو للمعالجة"),
        cvd_type: int = Form(..., description="نوع عمى الألوان: 1=protanopia, 2=deuteranopia, 3=tritanopia"),
        severity: float = Form(1.0, ge=0.0, le=1.0, description="شدة التصحيح (من 0.0 إلى 1.0)")
):
    if cvd_type not in cvd_types:
        raise HTTPException(status_code=400, detail="نوع عمى الألوان غير صالح.")

    # <<< الجزء الذكي: التحقق من نوع الملف
    content_type = file.content_type
    print(f"Detected content type: {content_type}")  # لطباعة النوع أثناء الاختبار

    if content_type.startswith("image/"):
        return await handle_image_correction(file, cvd_types[cvd_type], severity)

    elif content_type.startswith("video/"):
        return await handle_video_correction(file, cvd_types[cvd_type], severity)

    else:
        raise HTTPException(
            status_code=415,  # 415 Unsupported Media Type
            detail=f"نوع الملف '{content_type}' غير مدعوم. يرجى رفع صورة أو فيديو."
        )
