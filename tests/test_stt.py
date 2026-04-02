"""Minimal WebRTC test client: streams mic audio to the server and prints STT results."""

import asyncio
import json
import signal

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

SERVER_URL = "http://localhost:8080/offer"
# avfoundation default mic on macOS; swap to "hw:0" with "alsa" format on Linux
MIC_SOURCE = ":default:"
MIC_FORMAT = "avfoundation"


async def run() -> None:
    player = MediaPlayer(MIC_SOURCE, format=MIC_FORMAT)
    pc = RTCPeerConnection()

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"[data channel open: {channel.label}]")

        @channel.on("message")
        def on_message(msg):
            try:
                data = json.loads(msg)
            except Exception:
                return
            event = data.get("event", "")
            if event == "stt_result":
                print(f"STT: {data['text']}")
            elif event == "vad_utterance":
                print(f"  [VAD utterance {data['duration']}s]")
            elif event == "stt_empty":
                print("  [STT: empty]")
            elif event == "pipeline_error":
                print(f"  [error: {data.get('error')}]")

    @pc.on("connectionstatechange")
    async def on_state_change():
        print(f"[connection: {pc.connectionState}]")

    audio_track = player.audio
    if audio_track is None:
        raise RuntimeError("No audio track from MediaPlayer — check mic source/format")
    pc.addTrack(audio_track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            SERVER_URL,
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        ) as resp:
            answer_data = await resp.json()

    answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
    await pc.setRemoteDescription(answer)

    print("Connected. Speak into your mic. Ctrl+C to quit.")

    stop = asyncio.get_event_loop().create_future()
    asyncio.get_event_loop().add_signal_handler(signal.SIGINT, stop.set_result, None)
    await stop

    await pc.close()
    print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(run())
