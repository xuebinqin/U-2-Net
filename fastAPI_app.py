'''
FastAPI App
'''
from io import BytesIO
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import Response
import traceback
from segment_image import remove
from PIL import Image

app = FastAPI(title='BG Removal API')

@app.post('/remove-bg')
async def remove_background(image: UploadFile, post_process:float = Form(default = False)):
    """
    Remove Background from an image
    """
    try:
        image = await image.read()
        buffer = BytesIO(image)
    except Exception as e:
        e = traceback.format_exc()
        raise HTTPException(status_code=420, detail=f"Image loading error :: {e}")

    try:
        data = remove(Image.open(buffer), bool(post_process))
        print(type(data))
        return Response(content=data, media_type="image/png")
    except Exception as e:
        e = traceback.format_exc()
        raise HTTPException(status_code=420, detail=f"Segmentation Error:: {e}")

if __name__ == "__main__":
    uvicorn.run("fast_app:app", reload=True, debug = True, host = '0.0.0.0', port = 5678)