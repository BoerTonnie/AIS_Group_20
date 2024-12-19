import asyncio
import websockets
import json

URL = "http://192.168.1.193:8080/"  # Replace with your WebSocket URL

async def live_data():
    async with websockets.connect(URL) as websocket:
        while True:
            data = await websocket.recv()
            parsed_data = json.loads(data)
            print(parsed_data)  # Print or process your live data here

asyncio.run(live_data())