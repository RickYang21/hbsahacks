# Grandma Memory Bot — 60-second setup

Family members text photos to a bot over iMessage. Claude extracts tags/summary/people/era, rows land in Supabase, a live dashboard shows them, and a "Start Session" button kicks off a reminiscence-therapy iMessage conversation with grandma (Person B's half of the project owns the conversation logic).

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Create a Supabase project. Paste `schema.sql` into the SQL editor and run it. Then in the SQL editor, run this to create the public storage bucket:
   ```sql
   insert into storage.buckets (id, name, public) values ('memory-photos', 'memory-photos', true);
   ```
4. Copy `.env.example` → `.env` and fill in:
   - `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` — from Supabase → Settings → API → `service_role`
   - `ANTHROPIC_API_KEY`
   - `BLUEBUBBLES_URL` — e.g. `http://localhost:1234` if this runs on the same Mac as BlueBubbles, otherwise the Mac's LAN IP
   - `BLUEBUBBLES_PASSWORD` — from the BlueBubbles server settings
5. Update the seeded phone numbers in `schema.sql` to real iMessage-registered numbers (grandma's iPhone + your family tester phone), or run a follow-up `update` in Supabase.
6. `uvicorn main:app --reload --port 8000`
7. In another terminal: `ngrok http 8000` — copy the `https://…ngrok…` URL.
8. In BlueBubbles Server → Settings → API & Webhooks → add webhook `{ngrok-url}/webhook/bluebubbles`, enable the `new-message` event.
9. Open http://localhost:8000 for the dashboard.
10. Test ingest: from the family tester phone, iMessage the BlueBubbles Mac's address with a photo + caption like `Mom in her rose garden, 1987.`
11. Expected: the dashboard shows the new memory within ~2s, and the family tester phone receives a warm confirmation reply.

## File map

- `main.py` — FastAPI app: BlueBubbles webhook, family handler, grandma stub, dashboard API, session start.
- `bluebubbles.py` — BlueBubbles REST client (`send_text`, `send_image`, `download_attachment`) + inbound payload parser.
- `dashboard.html` — single-file Tailwind + vanilla JS dashboard (served at `/`, polls every 2s).
- `schema.sql` — DB schema shared with Person B. Seeds Margaret + Sarah.
- `requirements.txt`, `.env.example`, `.gitignore`.

## Contract with Person B

Person B writes `sessions`, `turns` (with `role='grandma'` for her replies, `role='bot'` for bot replies), and `grandma_profile_facts`. The dashboard reads all three. Do not rename any columns in `schema.sql`.

The inbound BlueBubbles webhook already routes grandma-sourced messages to `handle_grandma()` in `main.py` — that function is a stub marked `# TODO: Person B implements here`.

## Simulate a BlueBubbles webhook without a phone

```bash
curl -X POST http://localhost:8000/webhook/bluebubbles \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "new-message",
    "data": {
      "guid": "iMessage;-;+15550000002;FAKE",
      "text": "Mom in her rose garden, 1987",
      "isFromMe": false,
      "handle": { "address": "+15550000002" },
      "chats": [{ "guid": "iMessage;-;+15550000002" }],
      "attachments": [
        { "guid": "REAL_ATTACHMENT_GUID", "mimeType": "image/jpeg", "transferName": "IMG_0001.jpeg" }
      ]
    }
  }'
```

Note: the `attachments[].guid` must be a real guid that BlueBubbles can serve via `/api/v1/attachment/{guid}/download`, because the family handler downloads the bytes. To fully dry-run without BlueBubbles, leave `attachments: []` and a plain text message — it will be stashed in the 60s merge buffer.
