# token_gen.py
from livekit import api
import os

API_KEY = "devkey"
API_SECRET = "secret"

token = (
    api.AccessToken(API_KEY, API_SECRET)
        .with_identity("voice-bot")
        .with_name("Voice Bot")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="testroom",
            )
        )
        .to_jwt()
)

print("Token:", token)
# Token: 
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiVm9pY2UgQm90IiwidmlkZW8iOnsicm9vbUpvaW4iOnRydWUsInJvb20iOiJ0ZXN0cm9vbSIsImNhblB1Ymxpc2giOnRydWUsImNhblN1YnNjcmliZSI6dHJ1ZSwiY2FuUHVibGlzaERhdGEiOnRydWV9LCJzdWIiOiJ2b2ljZS1ib3QiLCJpc3MiOiJkZXZrZXkiLCJuYmYiOjE3NTY5NDg4NjMsImV4cCI6MTc1Njk3MDQ2M30.Fd9-imo1hS_SjKx8i7oUMA6re4IYbuuRtJV8fG_7h1U