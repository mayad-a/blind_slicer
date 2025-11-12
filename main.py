# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io
from daltonlens import simulate

app = FastAPI(
    title="Color Correction API",
    description="واجهة برمجية لتصحيح الصور لمرضى عمى الألوان."
)

# ربط الأرقام بأنواع عمى الألوان
cvd_types = {
    1: simulate.Deficiency.PROTAN,   # protanopia
    2: simulate.Deficiency.DEUTAN,   # deuteranopia
    3: simulate.Deficiency.TRITAN,   # tritanopia
}

simulator = simulate.Simulator_Machado2009()

# دالة التصحيح مع تأثير الشدة
def daltonize_with_severity(original, simulated, severity=1.0):
    error = original.astype(int) - simulated.astype(int)
    corrected = original.astype(int) + (error * severity)
    corrected = np.clip(corrected, 0, 255).astype("uint8")
    return corrected

# POST endpoint لتصحيح الصور
@app.post("/correct_image")
async def correct_image(
    file: UploadFile = File(...),
    cvd_type: int = Form(..., description="نوع عمى الألوان: 1=protanopia, 2=deuteranopia, 3=tritanopia"),
    severity: float = Form(1.0, ge=0.0, le=1.0, description="شدة التصحيح (من 0.0 إلى 1.0)")
):
    if cvd_type not in cvd_types:
        raise HTTPException(
            status_code=400,
            detail="نوع عمى الألوان غير صالح. استخدم: 1=protanopia, 2=deuteranopia, 3=tritanopia"
        )

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.asarray(img)
    except Exception:
        raise HTTPException(status_code=400, detail="ملف الصورة غير صالح.")

    # المحاكاة مع الشدة
    simulated = simulator.simulate_cvd(img_np, cvd_types[cvd_type], severity=severity)

    # التصحيح مع تأثير الشدة
    corrected = daltonize_with_severity(img_np, simulated, severity)
    corrected_img = Image.fromarray(corrected)

    buf = io.BytesIO()
    corrected_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
