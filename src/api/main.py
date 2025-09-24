import io
import logging
import sys
import warnings

import uvicorn
from fastapi import FastAPI, File, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.append("../PhotoDocsCreator")

from PhotoDocsCreator import PhotoDocsCreator

logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return FileResponse("static/index.html")


@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))
    try:
        output_image = pdc.process(input_image)
    except Exception as ex:
        logger.info(ex)

    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    return Response(
        content=img_byte_arr,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=processed.png"},
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pdc = PhotoDocsCreator()
    uvicorn.run(app, host="0.0.0.0", port=8000)
