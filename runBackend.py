import os
from threading import Thread

RUN_COMMAND = "uvicorn ai_disease_prediction_backend.api.main:app"

# os.system("uvicorn ai_disease_prediction_backend.api.main:app --reload")

def run():
    os.system(RUN_COMMAND)
    pass

thrd = Thread(target=run)

thrd.start()
thrd.join()