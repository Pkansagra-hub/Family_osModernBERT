# Operation Ontology v1

This document defines canonical operation families and accepted aliases used for retrieval and reranking.

## Canonical families

- **analyze**: analyse, analyze, classify, score
- **approve**: accept, approve, authorize, validate
- **archive**: archive, retire, store
- **complete**: close, complete, finish, resolve
- **create**: add, append, copy, create, draft, generate, initiate, log, new, open, record
- **delete**: cancel, delete, drop, remove
- **export**: export, extract
- **finance**: balance, claim, expense, invoice, payment, reconcile, refund
- **monitor**: monitor, observe, track, watch
- **retrieve**: check, download, fetch, find, get, list, lookup, read, retrieve, view
- **schedule**: book, plan, reschedule, schedule, set
- **search**: discover, find, query, search
- **send**: dispatch, forward, invite, message, notify, push, reply, send, share, submit
- **summarize**: digest, summarize, summary
- **update**: assign, change, edit, merge, modify, move, patch, pin, rename, reopen, tag, update
- **upload**: attach, ingest, upload

## Verb to canonical mapping

- `accept` -> `approve`
- `add` -> `create`
- `append` -> `create`
- `approve` -> `approve`
- `archive` -> `archive`
- `assign` -> `update`
- `cancel` -> `delete`
- `check` -> `retrieve`
- `close` -> `complete`
- `complete` -> `complete`
- `copy` -> `create`
- `create` -> `create`
- `decline` -> `update`
- `delete` -> `delete`
- `download` -> `retrieve`
- `draft` -> `create`
- `export` -> `export`
- `find` -> `search`
- `forward` -> `send`
- `generate` -> `create`
- `get` -> `retrieve`
- `invite` -> `send`
- `list` -> `retrieve`
- `log` -> `create`
- `merge` -> `update`
- `move` -> `update`
- `pin` -> `update`
- `reconcile` -> `finance`
- `record` -> `create`
- `refill` -> `update`
- `refund` -> `finance`
- `remove` -> `delete`
- `rename` -> `update`
- `reopen` -> `update`
- `reply` -> `send`
- `schedule` -> `schedule`
- `search` -> `search`
- `send` -> `send`
- `set` -> `schedule`
- `share` -> `send`
- `submit` -> `send`
- `tag` -> `update`
- `update` -> `update`
- `upload` -> `upload`

## Explicit sub-cluster overrides

- `accept_invite` -> `approve`
- `add_reminder` -> `create`
- `cancel_scheduled_message` -> `delete`
- `create_event` -> `create`
- `create_template` -> `create`
- `decline_invite` -> `update`
- `delete_event` -> `delete`
- `delete_template` -> `delete`
- `find_free_slots` -> `search`
- `get_delivery_status` -> `retrieve`
- `get_event` -> `retrieve`
- `invite_attendees` -> `send`
- `list_events` -> `retrieve`
- `list_messages` -> `retrieve`
- `schedule_message` -> `schedule`
- `send_push` -> `send`
- `send_sms` -> `send`
- `send_whatsapp` -> `send`
- `update_event` -> `update`
- `update_template` -> `update`
