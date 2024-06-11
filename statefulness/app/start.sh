# cd to root of this file
cd "$(dirname "$0")"

# install dependencies
pip install fastapi
pip install uvicorn

# run the app
python3 server.py