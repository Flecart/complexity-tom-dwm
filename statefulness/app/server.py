import fastapi
import json
from pydantic import BaseModel, RootModel
app = fastapi.FastAPI()

@app.get("/")
def read_root():
    # return static html page
    return fastapi.responses.FileResponse("index.html")

@app.get("/data/{item_id}")
def read_item(item_id: str):
    with open(f"../data/{item_id}.json", "r") as f:
        data = json.load(f)

    return data

class Sample(BaseModel):
    prompt: str
    question: str
    answer: str
    num_states: int = -1
    num_highlights: list[tuple[int, int]] = []

class SampleList(BaseModel):
    data: list[Sample]

@app.post("/data/{item_id}")
def write_item(item_id: str, data: SampleList):
    with open(f"../data/{item_id}.json", "w") as f:

        jsons = [x.model_dump() for x in data.data]
        json.dump(jsons, f, indent=4)

    return {"status": "ok"}

# uvicorn serve

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")