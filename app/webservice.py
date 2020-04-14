import sys, os
from pathlib import Path
import aiohttp
import aiofiles
import asyncio
import numpy as np
import pickle
import base64
from io import BytesIO

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response, JSONResponse, RedirectResponse
from starlette.routing import Route, Mount
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from fastai.vision import *

export_file_url = 'https://pokedexproject.s3.eu-west-2.amazonaws.com/export.pkl'
export_file_name = 'export.pkl'

path = Path(__file__).parent

templates = Jinja2Templates(directory= path / 'static/templates')
static_files = StaticFiles(directory= path / 'static')

print('Initialising Learner...')

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)

async def setup_learner():
    print('Downloading model + parameters')
    await download_file(export_file_url, path / export_file_name)
    print('Done.')
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


#learn = load_learner(path / 'models','export.pkl')
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


with open(path / 'cards.pkl','rb') as f:
    cards = pickle.load(f)

async def home(request):
    return templates.TemplateResponse('index.html', {'request': request})

async def isup(request):
	return Response("True", status_code=200, media_type='text/plain')

async def card(request):
    if request.path_params['name'] == 'notknown':
        return templates.TemplateResponse('cardnotknown.html', {'request': request})

    info = cards.get(request.path_params['name'],None)
    return templates.TemplateResponse('card.html', {'request': request, 'card':info})

async def feedback(request):
    ans,name = request.query_params['ans'],request.query_params['name']
    with open('feedback.csv','a') as f:
        if ans == 'correct':
            f.write(f"{name},1\n")
        elif ans == 'wrong':
            f.write(f"{name},0\n")
        else:
            pass
    return RedirectResponse(url=f"/", status_code=303)


async def analyze(request):
    img_data = await request.form()
    img = open_image(img_data['file'].file)
    print('Got image, making prediction...')


    predicted_class, pred_class_i, class_probs = learn.predict(img)

    info = cards.get(str(predicted_class),None)
    if info:
        return RedirectResponse(url=f"/card/{info['sname']}", status_code=303)
    else:
        return RedirectResponse(url="/card/notknown")


routes = [
    Route('/', home),
    Route('/isup', isup),
    Route('/card/{name}', card),
    Route('/analyze', analyze, methods=['POST']),
    Route('/feedback', feedback),
    Mount('/static', static_files, name='static'),

]

app = Starlette(debug=True, routes=routes)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=80, log_level="info")


