from __future__ import annotations

from typing import Any

import imapclient
import json
import logging
import time


from email import message_from_bytes
from email.policy import default
from email.utils import parseaddr

from peteos.oap.agentic_object import AgenticObject


from petekit.text.buffer_manager import Buffer, BufferManager, HookPoint


_logger = logging.getLogger("imaper")

_IMAPER_SCOPE = "email:"
_IMAPER_DOC_BUFFER = "doc:imap"
_IMAPER_CONFIG_BUFFER = _IMAPER_SCOPE + "config"
_IMAPER_STATUS_BUFFER = _IMAPER_SCOPE + "status"
_IMAPER_TREE_BUFFER = _IMAPER_SCOPE + "tree"
_IMAPER_MAILBOX_SCOPE = _IMAPER_SCOPE + "box"
_IMAPER_DOC = f"""\
## IMAPer

The IMAPer manages one IMAP account for receiving emails via a buffer-based API.

### Buffers

- **{_IMAPER_CONFIG_BUFFER}**  --  holds IMAP credentials and server settings
- **{_IMAPER_STATUS_BUFFER}**  --  holds runtime status such as IMAP connection state, folder count, and last refresh time
- **{_IMAPER_TREE_BUFFER}**  --  holds the mailbox tree as nested JSON (e.g. `{{"INBOX": {{"Sent": {{}}, "Work": {{}}}}}}`)
- **{_IMAPER_MAILBOX_SCOPE}/INBOX/envelopes**  --  holds the list of emails in INBOX with metadata (sender, recipient, subject, date, flags, uid)
- **{_IMAPER_MAILBOX_SCOPE}/INBOX/Sent/envelopes**  --  same for a subfolder; path segments use "/" as hierarchy separator
- **{_IMAPER_MAILBOX_SCOPE}/<mailbox>/<uid>.body**  --  the plain-text body of a single email; referenced from the `body` field in the envelopes list
- **{_IMAPER_MAILBOX_SCOPE}/<mailbox>/envelopes.filter**  --  optional; when present, narrows the envelopes list with server-side search criteria

### Configuration

To configure the IMAP client, write to `{_IMAPER_CONFIG_BUFFER}` with a JSON object containing:

```json
{{
  "host": "imap.example.com",
  "port": 993,
  "username": "your-username",
  "password": "your-password",
  "timeout": 15
}}
```

All fields are optional. The `timeout` field controls connection timeout in seconds (defaults to 15).

The IMAPer automatically connects when the config is written and reconnects on any credential change. It tries SSL first; if that fails, it falls back to a plain IMAP4 connection.

### Refreshing the Mailbox Tree

The `{_IMAPER_TREE_BUFFER}` buffer auto-refreshes from the server on every read. The agent does not need to manage a trigger buffer  --  reading the tree is sufficient to keep it current.

### Refreshing a Single Mailbox Email List

The `{_IMAPER_MAILBOX_SCOPE}/<mailbox>/envelopes` buffer auto-refreshes from the server on every read. The agent does not need to manage a trigger buffer  --  reading the envelope buffer is sufficient to keep it current.

The response includes a `total` field: the count of all emails matching the current filter (before pagination). This lets the agent know how many pages exist.

### Reading a Single Email Body

The email body is stored in a separate buffer and is not included in the `envelopes` list. The `body` field in each envelope entry is a reference string pointing to the body buffer: `{_IMAPER_MAILBOX_SCOPE}/<mailbox>/<uid>.body`. To read the full message text, the agent reads the body buffer directly by that path.

### Reading a Single Email (Full Flow)

Reading from the IMAPer follows a strict chain  --  each step refreshes from the server and prepares the next:

1. Read `{_IMAPER_TREE_BUFFER}`  --  fetches the mailbox list from the server
2. Read `{_IMAPER_MAILBOX_SCOPE}/<mailbox>/envelopes`  --  fetches the UID list for that mailbox and creates empty UID buffers; each entry has a `body` field with the body buffer path
3. Read `{_IMAPER_MAILBOX_SCOPE}/<mailbox>/<uid>`  --  fetches the full RFC822 message and writes the body to `<uid>.body`; the `body` field in the response holds the body buffer path
4. Read the path stored in the `body` field  --  the plain-text body text itself

### Filtering and Paginating the Email List

To narrow the email list by sender, subject, flags, or date, create a filter buffer at `{_IMAPER_MAILBOX_SCOPE}/<mailbox>/envelopes.filter`. When this buffer exists, its content is read on every `envelopes` refresh and applied as the search criteria. All criteria are AND-ed together  --  a message must match every criterion to be included.

```json
{{
  "criteria": [
    {{"key": "UNSEEN"}},
    {{"key": "SUBJECT", "value": "hello"}}
  ],
  "limit": 20,
  "offset": 40
}}
```

Available filter criteria keys: `ALL`, `UNSEEN`, `SEEN`, `FLAGGED`, `UNFLAGGED`, `ANSWERED`, `UNANSWERED`, `DELETED`, `DRAFT`, `FROM`, `TO`, `CC`, `SUBJECT`, `BODY`, `TEXT`, `BEFORE`, `SINCE`, `ON`.

Delete the filter buffer to return to the unfiltered full list. The `limit` is optional; if absent, all matching emails are returned. Default `offset` is 0.

### Updating Flags on a UID Buffer

The agent can update the IMAP flags on a single email by writing to its UID buffer.
Only the `flags` field can be changed  --  the sender, recipient, subject, date,
uid, and body fields are immutable. Attempts to change any other field are
rejected by the guard hook.

To mark a message as read:

```json
{{
  "uid": 15,
  "sender": "alice@example.com",
  "recipient": "bob@example.com",
  "subject": "Hello",
  "date": "Thu, 29 Jan 2026 16:25:27 +0000",
  "flags": ["\\\\Seen", "\\\\Answered"],
  "body": "email:box/INBOX/15.body"
}}
```

The BEFORE_UPDATE_BUFFER guard validates that only the `flags` field changed,
computes the delta, and rejects the update if anything else was modified.

The AFTER_UPDATE_BUFFER hook applies the flag deltas to the IMAP server
using `add_flags([uid], ['\\\\Seen'])` and `remove_flags([uid], ['\\\\Seen'])` so
the changes are reflected on the server immediately.

Available IMAP flags:

- `\\Seen`  --  message has been read
- `\\Answered`  --  message has been replied to
- `\\Flagged`  --  message is flagged/starred
- `\\Draft`  --  message is a draft
- `\\Deleted`  --  marked for deletion (removed on expunge)
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Util                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #

def _get_imap_config(config_buffer: Buffer | None) -> dict:
    """Read and parse the email:config buffer contents as JSON.

    Returns an empty dict if the buffer is None (e.g. on first creation).
    Raises json.JSONDecodeError on invalid JSON.
    """
    if config_buffer is None:
        return {}
    # Parse and return the config as a dict.
    return json.loads(config_buffer.get_text())


def _imap_connection_changed(old_config: dict, new_config: dict) -> bool:
    """Return True if any IMAP connection credential has changed.

    Compares host, port, username, and password fields between the two configs.
    A change in any of these fields signals that the IMAP connection must be
    re-established.
    """
    # Check each credential field for a difference.
    if old_config.get("host") != new_config.get("host"):
        return True
    if old_config.get("port") != new_config.get("port"):
        return True
    if old_config.get("username") != new_config.get("username"):
        return True
    if old_config.get("password") != new_config.get("password"):
        return True

    return False


def _parse_mailbox_list(raw_response: tuple) -> dict[str, str | list[list[str]]] | None:
    """Extract mailbox names and delimiter from an IMAP4.list response tuple.

    The raw response is a (status, entries) tuple. Each entry is either:
    - imaplib: a bytes string of the form: b'(\\HasNoChildren, ".") "INBOX/Work"'
    - imapclient: a tuple of (flags, delimiter, name) where name is already unicode

    The delimiter is extracted from the first entry that carries one. Each
    mailbox name is split by the delimiter into its path segments.

    Returns a dict with "delimiter" (the server's hierarchy char) and "mailboxes"
    (list of path lists, e.g. [["INBOX"], ["INBOX", "Work"]]). Returns None if
    the server response is not OK.
    """
    # Reject non-OK responses  --  the command failed on the server side.
    if not raw_response or raw_response[0] != 'OK':
        _logger.error(f"[IMAPer] mailbox list response not OK: {raw_response[0] if raw_response else 'empty'}")
        return None

    # Pull the entries list from the IMAP response  --  skip on malformed response.
    entries = raw_response[1] if len(raw_response) > 1 else []

    # Pull the delimiter from the first entry  --  imapclient returns it as bytes at index 1.
    # Format per entry: ((flags,...), delimiter_bytes, name_str)
    # Set a default delimiter in case the server response is malformed.
    delimiter = "/"
    if entries:
        # Extract the raw delimiter bytes from the first entry.
        raw_delimiter = entries[0][1]
        if isinstance(raw_delimiter, bytes):
            # Decode the bytes delimiter to a string for later string operations.
            try:
                delimiter = raw_delimiter.decode("utf-8")
            except Exception:
                pass

    # Extract each mailbox name  --  entry[2] is the full path as a str already.
    # Split it by the delimiter to get the individual path segments.
    names = []
    for entry in entries:
        # Skip malformed entries  --  need a 3-tuple of (flags, delimiter, name).
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue

        # Pull the name from index 2, split it by the hierarchy delimiter, and record the path.
        name = str(entry[2]) if entry[2] else ""
        if name:
            path = name.split(delimiter) if delimiter else [name]
            names.append(path)

    return {"delimiter": delimiter, "mailboxes": names}


def _build_mailbox_tree(paths: list[list[str]]) -> dict:
    """Build a nested dictionary tree from a list of mailbox path lists.

    Each path list like ["INBOX", "Work", "Work"] becomes a nested dict chain.
    The resulting tree uses empty dicts as leaf placeholders.

    Example:
        [["INBOX"], ["INBOX", "Sent"], ["INBOX", "Work", "Work"]]
        becomes
        {"INBOX": {"Sent": {}, "Work": {"Work": {}}}}.
    """
    # Start with an empty root  --  the trunk of the mailbox tree.
    root: dict = {}

    # Walk each path and hang it off the root or an existing branch.
    for path in paths:
        # Begin at the root for every path  --  descend from here.
        node = root

        # Walk each segment of the current path and build out the tree.
        for segment in path:
            # Create a new branch if this segment has not been seen yet.
            if segment not in node:
                node[segment] = {}

            # Step into the segment for the next iteration.
            node = node[segment]

    # The fully built tree is returned  --  empty dicts mark leaf nodes.
    return root


def _fetch_mailbox_status(imap_connection: imapclient.IMAPClient, mailbox_name: str) -> dict | None:
    """Return the SELECT metadata for a mailbox without fetching the email list.

    Uses imapclient's select_folder which returns a clean dictionary with SELECT
    response fields. Returns None on any failure (logged).

    Returns:
        {
            "exists": int,
            "uidvalidity": int,
            "flags": list[str],
            "recent": int,
            "unseen": int,
        }
    """
    # Reject if there is no active IMAP connection.
    if imap_connection is None:
        raise RuntimeError("no active IMAP connection")

    # Select the mailbox  --  imapclient returns a clean dict, not raw bytes.
    try:
        result = imap_connection.select_folder(mailbox_name)
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] SELECT failed for {mailbox_name}: {e}")
        return None

    # Bail out if the result is empty or invalid.
    if not result:
        _logger.error(f"[IMAPer] SELECT returned empty for {mailbox_name}")
        return None

    # Extract the metadata from the imapclient response dictionary.
    # imapclient returns byte keys: b'EXISTS', b'UIDVALIDITY', b'FLAGS', etc.
    metadata: dict = {
        "exists": int(result.get(b'EXISTS', [0])[0]) if isinstance(result.get(b'EXISTS'), (list, tuple)) else int(result.get(b'EXISTS', 0)),
        "uidvalidity": int(result.get(b'UIDVALIDITY', [0])[0]) if isinstance(result.get(b'UIDVALIDITY'), (list, tuple)) else int(result.get(b'UIDVALIDITY', 0)),
        "flags": list(result.get(b'FLAGS', [])) if isinstance(result.get(b'FLAGS'), (list, tuple)) else [],
        "recent": int(result.get(b'RECNET', [0])[0]) if isinstance(result.get(b'RECNET'), (list, tuple)) else int(result.get(b'RECNET', 0)),
        "unseen": int(result.get(b'UNSEEN', [0])[0]) if isinstance(result.get(b'UNSEEN'), (list, tuple)) else int(result.get(b'UNSEEN', 0)),
    }

    return metadata


def _list_mailbox_emails(
    imap_connection: imapclient.IMAPClient,
    mailbox_name: str,
    filter: dict | None = None,
) -> dict | None:
    """Return the list of all emails in a mailbox with ENVELOPE and FLAGS metadata.

    If a filter dict is provided, the search is narrowed to match the criteria
    before fetching the email list. The default limit is 50 emails at offset 0.

    Args:
        filter: Optional filter dict with:
            - criteria: list of {"key": "...", "value": "..."} criteria
            - limit: int (default 50)
            - offset: int (default 0) -- page start position

    Returns:
        {
            "mailbox": str,
            "total": int,
            "uids": list[int],
            "emails": [
                {
                    "uid": int,
                    "sender": str,
                    "recipient": str,
                    "subject": str,
                    "date": str,
                    "flags": list[str],
                },
                ...
            ]
        }
    """
    # Reject if there is no active IMAP connection.
    if imap_connection is None:
        raise RuntimeError("no active IMAP connection")

    # Select the mailbox  --  imapclient handles state management.
    try:
        imap_connection.select_folder(mailbox_name)
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] SELECT failed for {mailbox_name}: {e}")
        return None

    # Build the IMAP search criteria from the filter dict.
    # Each criterion is {"key": "UNSEEN"} or {"key": "FROM", "value": "alice"}.
    imap_criteria: list = []
    if filter:
        for crit in filter.get("criteria", []):
            key = str(crit.get("key", "")).upper()
            value = str(crit.get("value", ""))
            if value:
                imap_criteria.append([key, value])
            else:
                imap_criteria.append([key])

    # Extract pagination parameters from the filter.
    # A non-positive or absent limit means unlimited  --  no slice is applied.
    limit = int(filter.get("limit", 0) if filter else 0)
    offset = int(filter.get("offset", 0) if filter else 0)
    if offset < 0:
        offset = 0

    # Search with the given criteria, or ALL if no filter is provided.
    try:
        if imap_criteria:
            message_ids = imap_connection.search(imap_criteria)
        else:
            message_ids = imap_connection.search(["ALL"])
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] SEARCH failed for {mailbox_name}: {e}")
        return None

    # Record the total count of matching emails before pagination slicing.
    total = len(message_ids)

    # Handle the empty mailbox case  --  return an empty list cleanly.
    if not message_ids:
        return {"mailbox": mailbox_name, "total": 0, "uids": [], "emails": []}

    # Apply pagination window  --  only when a positive limit was provided.
    if limit > 0:
        message_ids = message_ids[offset:offset + limit]

    # Fetch full message and FLAGS for all message IDs in one batch.
    # RFC822 is the full raw message bytes, parsed by the email library.
    try:
        result = imap_connection.fetch(message_ids, ["BODY.PEEK[]", "FLAGS"])
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] FETCH failed for {mailbox_name}: {e}")
        return None

    # Parse each email header with the email library  --  all decoding is automatic.
    emails: list = []
    for uid, fields in result.items():
        raw_headers = fields.get(b'RFC822')
        if raw_headers and isinstance(raw_headers, bytes):
            msg = message_from_bytes(raw_headers, policy=default)
            # parseaddr returns (name, addr)  --  addr is the bare email address.
            if msg["From"]:
                _, sender = parseaddr(msg["From"])
            else:
                sender = ""
            if msg["To"]:
                _, recipient = parseaddr(msg["To"])
            else:
                recipient = ""
            subject = str(msg["Subject"]) if msg["Subject"] else ""
            date = str(msg["Date"]) if msg["Date"] else ""
        else:
            sender = ""
            recipient = ""
            subject = ""
            date = ""

        # FLAGS  --  server-side read/answered/etc. status.
        raw_flags = fields.get(b'FLAGS', [])
        flags = [f.decode("utf-8") if isinstance(f, bytes) else str(f) for f in raw_flags]

        emails.append({
            "uid": uid,
            "sender": sender,
            "recipient": recipient,
            "subject": subject,
            "date": date,
            "flags": flags,
        })

    # Return the complete mailbox email list with the total match count.
    return {
        "mailbox": mailbox_name,
        "total": total,
        "uids": message_ids,
        "emails": emails,
    }


def _mailbox_to_envelopes_buffer_name(mailbox_name: str) -> str:
    """Convert a mailbox path to its corresponding email:box envelope buffer name.

    INBOX          ->  email:box/INBOX/envelopes
    INBOX/Sent     ->  email:box/INBOX/Sent/envelopes
    """
    return _IMAPER_MAILBOX_SCOPE + f"/{mailbox_name}/envelopes"


def _fetch_email_by_uid(imap_connection: imapclient.IMAPClient, mailbox_name: str, uid: int) -> dict | None:
    """Fetch a single email by UID from the server.

    Returns a dict with uid, sender, recipient, subject, date, flags, and body.
    Returns None on any failure (logged).
    """
    # Reject if there is no active IMAP connection.
    if imap_connection is None:
        raise RuntimeError("no active IMAP connection")

    # Select the mailbox  --  required before fetching message data.
    try:
        imap_connection.select_folder(mailbox_name)
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] SELECT failed for {mailbox_name}: {e}")
        return None

    # Fetch the full RFC 822 message  --  email library handles all parsing.
    try:
        result = imap_connection.fetch([uid], ["RFC822", "FLAGS"])
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] FETCH failed for UID {uid} in {mailbox_name}: {e}")
        return None

    # Bail out if the UID was not returned in the response.
    if not result or uid not in result:
        return None

    # Parse the raw RFC 822 message  --  email library handles CRLF, charset,
    # multipart, and all decoding.
    fields = result[uid]
    raw_message = fields.get(b'RFC822')
    if isinstance(raw_message, bytes):
        msg = message_from_bytes(raw_message, policy=default)
    else:
        _logger.error(f"[IMAPer] BODY[1] for UID {uid} is not bytes: {type(raw_message)}")
        return None

    # Extract sender and recipient via parseaddr  --  gives (name, addr) tuples.
    sender = parseaddr(msg["From"])[1] if msg["From"] else ""
    recipient = parseaddr(msg["To"])[1] if msg["To"] else ""

    # Subject  --  str() on the Header object to get the decoded value.
    subject = str(msg["Subject"]) if msg["Subject"] else ""

    # Date from the header.
    date = msg["Date"] or ""

    # Flags as before  --  server-side read/answered/etc. status.
    raw_flags = fields.get(b'FLAGS', [])
    flags = [f.decode("utf-8") if isinstance(f, bytes) else str(f) for f in raw_flags]

    # Body  --  get_body returns the text/plain part, get_payload(decode=True)
    # decodes Content-Transfer-Encoding (quoted-printable, base64, etc).
    body_part = msg.get_body(("plain",))
    if body_part is not None:
        body_bytes = body_part.get_payload(decode=True)
        body = body_bytes.decode(body_part.get_content_charset() or "utf-8", errors="replace") if body_bytes else ""
    else:
        body = ""

    return {
        "uid": uid,
        "sender": sender,
        "recipient": recipient,
        "subject": subject,
        "date": date,
        "flags": flags,
        "body": body,
    }


def _flatten_tree(tree: dict) -> list[list[str]]:
    """Flatten a nested mailbox tree into a list of all leaf paths.

    Traverses the tree depth-first. Each leaf (an empty dict {}) becomes one
    path list  --  the list of keys from the root to that leaf.

    Example:
        {"INBOX": {"Sent": {}, "Work": {}}}
         ->  [["INBOX", "Sent"], ["INBOX", "Work"]]

    Returns an empty list if the tree is empty or not a dict.
    """
    result: list[list[str]] = []

    # Walk the tree recursively, building the path to each leaf.
    def walk(node: dict, path: list[str]) -> None:
        # Visit every key in the current node. Each key extends the current path.
        for key, value in node.items():
            current_path = path + [key]

            # If this key has children, descend into them before recording.
            if isinstance(value, dict):
                walk(value, current_path)

            # Every key marks a path  --  if it has children we recorded the deeper
            # path; if not, this is a leaf and this is the complete path.
            result.append(current_path)

    # Start the walk from the tree root with no prior path.
    if isinstance(tree, dict):
        walk(tree, [])

    return result


# -------------------------------------------------------------------------------------------------------------------- #
# Config hooks                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #

async def after_create_imap_config(
    imaper: IMAPer,
    config_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Register the validation and update hooks on the email:config buffer.

    Called once when email:config is created. Registers before_update_imap_config
    and after_update_imap_config directly on the buffer so the buffer owns them
    without needing pattern-based hook registration going forward.
    """
    # Wire validation and update hooks directly on the config buffer.
    config_buffer.register_hook(
        HookPoint.BEFORE_UPDATE_BUFFER,
        "imap_config_validated",
        before_update_imap_config,
    )
    config_buffer.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        "imap_config_updated",
        after_update_imap_config,
    )

    # Trigger the update hook to handle initial config setup.
    await after_update_imap_config(imaper, None, config_buffer, {})


def before_update_imap_config(
    imaper: IMAPer,
    old_config_buffer: Buffer | None,
    new_config_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Validate that the email:config buffer contains valid JSON.

    Returns None to allow the update, or an error string to block it.
    """
    # Reject non-JSON config  --  read directly from the incoming buffer.
    # At BEFORE_UPDATE_BUFFER the buffer text has not been updated yet,
    # so we must parse new_config_buffer.get_text() directly.
    try:
        _get_imap_config(new_config_buffer)
    except json.JSONDecodeError:
        return f"'{_IMAPER_CONFIG_BUFFER}' is not valid JSON"

    return None


async def after_update_imap_config(
    imaper: IMAPer,
    old_config_buffer: Buffer | None,
    new_config_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after email:config is created or updated. Triggered on first config setup.

    On creation old_config_buffer is None  --  this signals initial setup rather than a
    later config change.
    """
    # Parse both configs so the IMAPer can diff and react to changes.
    old_config = _get_imap_config(old_config_buffer)
    new_config = _get_imap_config(new_config_buffer)

    # React to credential changes  --  disconnect old connection if host/port/user/pass changed.
    if _imap_connection_changed(old_config, new_config):
        # Tear down the existing connection if one is alive.
        if imaper._imap_connection is not None:
            await imaper._disconnect_imap()

        # Establish a fresh connection with the new credentials.
        await imaper._connect_imap(
            new_config.get("host", ""),
            new_config.get("port", 993),
            new_config.get("username", ""),
            new_config.get("password", ""),
        )


# -------------------------------------------------------------------------------------------------------------------- #
# Buffer hooks                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def _before_update_filter(
    imaper: IMAPer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before an envelopes.filter buffer is updated.

    Validates that the filter JSON has the correct schema:
    {criteria: [{key: str, value: str}, ...], limit: int, offset: int}
    All fields are optional. Rejects with a description on any violation.
    """
    text = new_buffer.get_text()
    if not text or not text.strip():
        return None

    try:
        proposed = json.loads(text)
    except json.JSONDecodeError:
        return "filter buffer is not valid JSON"

    if not isinstance(proposed, dict):
        return "filter buffer must be a JSON object"

    # Validate criteria is a list of {key: str, value?: str} objects.
    if "criteria" in proposed:
        crit_list = proposed["criteria"]
        if not isinstance(crit_list, list):
            return "'criteria' must be a list"
        for i, crit in enumerate(crit_list):
            if not isinstance(crit, dict):
                return f"criteria[{i}] must be an object"
            key = crit.get("key", "")
            if not isinstance(key, str) or not key:
                return f"criteria[{i}].key must be a non-empty string"
            if "value" in crit and not isinstance(crit["value"], str):
                return f"criteria[{i}].value must be a string"

    # Validate limit is an int if present.
    if "limit" in proposed:
        if not isinstance(proposed["limit"], int):
            return "'limit' must be an integer"

    # Validate offset is an int if present.
    if "offset" in proposed:
        if not isinstance(proposed["offset"], int):
            return "'offset' must be an integer"

    return None


async def _after_update_filter(
    imaper: IMAPer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after an envelopes.filter buffer is updated.

    Extracts the mailbox name from the buffer name and triggers a refresh
    of the envelopes summary with the new filter criteria applied.
    """
    # Extract the mailbox name from the buffer name.
    # The buffer name is: _IMAPER_MAILBOX_SCOPE / mailbox_name / envelopes.filter
    parts = new_buffer.name.split("/")
    if len(parts) < 3 or parts[-1] != "envelopes.filter":
        return

    # Join the path segments between the scope and the "envelopes.filter" suffix.
    mailbox_name = "/".join(parts[1:-1])

    # Trigger the envelopes refresh with the new filter applied.
    try:
        await imaper._refresh_imap_box(mailbox_name)
    except Exception as e:
        _logger.error(f"[IMAPer] _after_update_filter failed for {mailbox_name}: {e}")


async def _after_drop_filter(
    imaper: IMAPer,
    old_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after an envelopes.filter buffer is dropped.

    Extracts the mailbox name from the buffer name and triggers a refresh
    of the envelopes summary with no filter  --  the full list is restored.
    """
    # Extract the mailbox name from the dropped buffer name.
    # The buffer name is: _IMAPER_MAILBOX_SCOPE / mailbox_name / envelopes.filter
    parts = old_buffer.name.split("/")
    if len(parts) < 3 or parts[-1] != "envelopes.filter":
        return

    # Join the path segments between the scope and the "envelopes.filter" suffix.
    mailbox_name = "/".join(parts[1:-1])

    # Trigger the envelopes refresh with no filter  --  restore the full list.
    try:
        await imaper._refresh_imap_box(mailbox_name)
    except Exception as e:
        _logger.error(f"[IMAPer] _after_drop_filter failed for {mailbox_name}: {e}")


async def _before_read_envelopes(
    imaper: IMAPer,
    buffer: Buffer,
    args: dict[str, Any],
) -> bool | str | None:
    """Called before an envelopes buffer is read.

    Fires on any email:box/<mailbox>/envelopes buffer. Extracts the mailbox name
    from the buffer name and triggers a refresh of the envelopes summary before
    the read proceeds. This keeps the email list current on every access.
    """
    # Extract the mailbox name from the buffer name.
    # The buffer name is: _IMAPER_MAILBOX_SCOPE / mailbox_name / envelopes
    # parts = ["email:box", "<mailbox>", ..., "envelopes"]
    parts = buffer.name.split("/")
    if len(parts) < 2 or parts[-1] != "envelopes":
        return True

    # Join the path segments between the scope and the "envelopes" suffix.
    mailbox_name = "/".join(parts[1:-1])

    # Refresh the envelopes summary for this mailbox.
    await imaper._refresh_imap_box(mailbox_name)

    return True


async def _before_read_imap_email(
    imaper: IMAPer,
    buffer: Buffer,
    args: dict[str, Any],
) -> bool | str | None:
    """Lazily fetch and populate a UID buffer on first read.

    Only fires on email:box/<mailbox>/<uid> buffers (pattern-matched).
    If the buffer already has content, proceeds without modification.
    If empty, fetches from the server and populates before allowing the read.
    """
    # Skip if the buffer already has content.
    if buffer.get_text():
        return True

    # Extract the UID from the buffer name  --  the last path segment is the UID.
    parts = buffer.name.split("/")
    uid = int(parts[-1])

    # Derive the mailbox name  --  the path between the mailbox scope and the UID.
    mailbox_name = "/".join(parts[1:-1])

    # Fetch the email from the server.
    email = _fetch_email_by_uid(imaper._imap_connection, mailbox_name, uid)
    if email is None:
        return f"[IMAPer] failed to fetch UID {uid} from {mailbox_name}"

    # Populate the buffer via set_text so hooks and references stay valid.
    # Store the body in a separate buffer and leave a reference in the JSON
    # so the agent can find it without parsing the full body text.
    body = email.pop("body", "")
    body_buffer_name = f"{buffer.name}.body"
    await imaper.create_buffer(body_buffer_name, text=body)

    # Put the body reference back so the agent sees where to find the body.
    email["body"] = body_buffer_name
    buffer.set_text(json.dumps(email, indent=2, ensure_ascii=False))

    return True


async def _after_update_imap_tree(
    imaper: IMAPer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after email:tree is updated.

    Fires whenever the tree buffer is written (via create_buffer with overwrite or write_buffer).
    Use this to drive any downstream state that depends on the current mailbox tree.
    On first creation old_buffer is None  --  this is initial setup, nothing to diff.
    """
    # Parse both trees from their buffer text. An empty or absent old buffer
    # means initial setup  --  there is nothing to diff against.
    try:
        old_tree = json.loads(old_buffer.get_text()) if old_buffer else {}
    except json.JSONDecodeError:
        return

    try:
        new_tree = json.loads(new_buffer.get_text())
    except json.JSONDecodeError:
        return

    # Flatten both trees to get the complete set of mailbox paths in each.
    old_paths = _flatten_tree(old_tree)
    new_paths = _flatten_tree(new_tree)

    # Convert each path list to the corresponding envelope buffer name.
    old_envelopes = {_mailbox_to_envelopes_buffer_name("/".join(p)) for p in old_paths}
    new_envelopes = {_mailbox_to_envelopes_buffer_name("/".join(p)) for p in new_paths}

    # Diff the two sets to find which mailboxes appeared and which vanished.
    added = new_envelopes - old_envelopes
    removed = old_envelopes - new_envelopes

    # Create an empty envelope buffer for every new mailbox in the tree.
    for buf_name in added:
        await imaper.create_buffer(buf_name, text="")

    # Drop the envelope buffer for every mailbox no longer in the tree.
    for buf_name in removed:
        await imaper.drop_buffer(buf_name)


async def _before_update_uid(
    imaper: IMAPer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before a UID buffer is updated.

    Guards that only the 'flags' field may change. Rejects any attempt to
    modify sender, recipient, subject, date, uid, or body.

    Extracts the flag diff and stores it in args for the AFTER hook to
    apply to the IMAP server.
    """
    # Reject updates on a buffer that has no prior content  --  use create not update.
    if old_buffer is None or not old_buffer.get_text():
        return "UID buffer has no prior content  --  use create instead of update"

    # Parse both the old and new buffer content as JSON.
    try:
        old_email = json.loads(old_buffer.get_text())
    except json.JSONDecodeError:
        return "UID buffer content is not valid JSON"

    try:
        new_email = json.loads(new_buffer.get_text())
    except json.JSONDecodeError:
        return "UID buffer content is not valid JSON"

    # The uid field may not change  --  the buffer name IS the UID.
    if old_email.get("uid") != new_email.get("uid"):
        return "uid field may not be changed  --  delete and recreate the UID buffer"

    # Build a list of fields that are allowed to change.
    immutable_fields = {"sender", "recipient", "subject", "date", "body", "uid"}

    # Check every field in the new buffer against the old  --  reject disallowed changes.
    for field, new_value in new_email.items():
        if field not in immutable_fields:
            continue
        old_value = old_email.get(field)
        if old_value != new_value:
            return f"field '{field}' may not be changed  --  only 'flags' can be updated"

    # Check for new fields that did not exist in the old buffer  --  reject those too.
    for field in new_email:
        if field not in immutable_fields:
            continue
        if field not in old_email:
            return f"field '{field}' may not be added  --  only 'flags' can be added"

    # Compute the flag diff between old and new.
    old_flags = set(old_email.get("flags", []))
    new_flags = set(new_email.get("flags", []))
    to_add = list(new_flags - old_flags)
    to_remove = list(old_flags - new_flags)

    # Store the deltas in args so the AFTER hook can apply them to the IMAP server.
    args["flags_to_add"] = to_add
    args["flags_to_remove"] = to_remove

    return None


async def _after_update_uid(
    imaper: IMAPer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after a UID buffer is updated.

    Applies the flag deltas stored by the BEFORE hook to the IMAP server
    via store(). Only called if the BEFORE hook returned None  --  flags
    are the only thing that changed.
    """
    # Extract the flag deltas that the BEFORE hook computed.
    to_add = args.get("flags_to_add", [])
    to_remove = args.get("flags_to_remove", [])

    # Nothing to do if no flag deltas were computed.
    if not to_add and not to_remove:
        return

    # Reject if there is no active IMAP connection.
    if imaper._imap_connection is None:
        return

    # Parse the new buffer to extract the UID and mailbox name.
    try:
        new_email = json.loads(new_buffer.get_text())
    except json.JSONDecodeError:
        return

    uid = new_email.get("uid")
    if not uid:
        return

    # Derive the mailbox name from the buffer name.
    # Buffer name: _IMAPER_MAILBOX_SCOPE / mailbox_name / uid
    parts = new_buffer.name.split("/")
    if len(parts) < 3:
        return
    mailbox_name = "/".join(parts[1:-1])

    # Select the mailbox first  --  required before store.
    try:
        imaper._imap_connection.select_folder(mailbox_name)
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] SELECT failed for {mailbox_name} in flag sync: {e}")
        return

    # Apply flags to add  --  use IMAP literal flag names.
    if to_add:
        try:
            imaper._imap_connection.add_flags([uid], to_add)
        except imapclient.Error as e:
            _logger.error(f"[IMAPer] add_flags failed for UID {uid}: {e}")

    # Apply flags to remove  --  use IMAP literal flag names.
    if to_remove:
        try:
            imaper._imap_connection.remove_flags([uid], to_remove)
        except imapclient.Error as e:
            _logger.error(f"[IMAPer] remove_flags failed for UID {uid}: {e}")


# -------------------------------------------------------------------------------------------------------------------- #
# IMAPer                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #

class IMAPer(BufferManager, AgenticObject):
    """."""

    _imap_connection: imapclient.IMAPClient | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._imap_connection = None

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_IMAPER_DOC_BUFFER, text=_IMAPER_DOC)

        # Seed the status buffer with an empty JSON object.
        self._create_buffer(_IMAPER_STATUS_BUFFER, text="{}")

        # Register the AFTER_UPDATE_BUFFER hook to drive envelope buffer lifecycle.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_update_imap_tree",
            _IMAPER_TREE_BUFFER,
            _after_update_imap_tree,
            apply_to_existing=True,
        )
        # Register the AFTER_CREATE_BUFFER hook to drive envelope buffer lifecycle.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_update_imap_tree",
            _IMAPER_TREE_BUFFER,
            lambda bm, new_buffer, args: _after_update_imap_tree(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the BEFORE_READ_BUFFER hook to refresh on every access.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_READ_BUFFER,
            "refresh_tree_on_read",
            _IMAPER_TREE_BUFFER,
            lambda *_: self._refresh_imap_mailbox_list(),
            apply_to_existing=True,
        )

        # Reject any attempt to drop the status buffer  --  it belongs to the IMAPer.
        status_buffer = self.get_buffer(_IMAPER_STATUS_BUFFER)
        if status_buffer is not None:
            status_buffer.register_hook(
                HookPoint.BEFORE_DROP_BUFFER,
                "imap_status_no_drop",
                lambda bm, buf, args: f"{_IMAPER_STATUS_BUFFER} cannot be dropped",
            )

        # Validate JSON on create, wire hooks on buffer once created.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "imap_config_json_check",
            _IMAPER_CONFIG_BUFFER,
            lambda bm, buf, args: before_update_imap_config(bm, None, buf, args),
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "imap_config_created",
            _IMAPER_CONFIG_BUFFER,
            after_create_imap_config,
            apply_to_existing=True,
        )

        # Register the lazy-fetch hook for UID buffers  --  fires before first read of each UID.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_READ_BUFFER,
            "lazy_fetch_imap_email",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/\d+$",
            _before_read_imap_email,
            apply_to_existing=True,
        )

        # Register flag sync hooks for UID buffers  --  guard allows only flags changes.
        # BEFORE: rejects updates that touch any field other than 'flags'.
        # AFTER: applies the flag deltas to the IMAP server via store().
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "uid_flags_guard",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/\d+$",
            _before_update_uid,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "uid_flags_sync",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/\d+$",
            _after_update_uid,
            apply_to_existing=True,
        )

        # Register the envelopes before-read hook  --  stub for future per-mailbox logic.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_READ_BUFFER,
            "before_read_envelopes",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/envelopes$",
            _before_read_envelopes,
            apply_to_existing=True,
        )

        # Register the filter validation and refresh hooks.
        # BEFORE: reject invalid filter schema.
        # AFTER: call _refresh_imap_box to apply the new filter to the envelopes list.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "before_update_filter",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/envelopes\.filter$",
            _before_update_filter,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_update_filter",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/envelopes\.filter$",
            _after_update_filter,
            apply_to_existing=True,
        )

        # Also fire the same filter hooks on CREATE  --  old_buffer is None on create.
        # CREATE hooks fire with 3 args  --  (bm, new_buffer, args).
        # UPDATE hooks expect 4 args  --  (bm, old_buffer, new_buffer, args).
        # The thin lambdas insert None for old_buffer to bridge the gap.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "before_create_filter",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/envelopes\.filter$",
            lambda bm, new_buffer, args: _before_update_filter(bm, None, new_buffer, args),
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_create_filter",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/envelopes\.filter$",
            lambda bm, new_buffer, args: _after_update_filter(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Refresh envelopes when the filter buffer is dropped  --  restore full list.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "after_drop_filter",
            rf"{_IMAPER_MAILBOX_SCOPE}/.+/envelopes\.filter$",
            _after_drop_filter,
            apply_to_existing=True,
        )

    async def _update_imap_connection(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
    ) -> dict:
        """Read the existing email:config, overlay the four credentials, write back if changed.

        Returns the new merged config dict.
        """
        # Read the existing config to preserve all other fields.
        old = _get_imap_config(self.get_buffer(_IMAPER_CONFIG_BUFFER))
        merged = {**old}
        merged["host"] = host
        merged["port"] = port
        merged["username"] = username
        merged["password"] = password

        # Only write to the buffer if at least one credential changed.
        if _imap_connection_changed(old, merged):
            await self.create_buffer(_IMAPER_CONFIG_BUFFER, text=json.dumps(merged, ensure_ascii=False), overwrite=True)

        return merged

    async def _connect_imap(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
    ) -> None:
        """Establish a new IMAP connection with the given credentials.

        Reads the current email:config buffer, overlays the four credential arguments,
        writes the merged result back to email:config as authoritative state.
        Uses imapclient for the IMAP connection with SSL and UID support.
        """
        # Update the config and get the merged result.
        config = await self._update_imap_connection(host, port, username, password)

        # Tear down any existing connection first.
        if self._imap_connection is not None:
            await self._disconnect_imap()

        # Read the timeout from config or default to 15 seconds.
        timeout = config.get("timeout", 15)
        if timeout is not None:
            timeout = int(timeout)

        # Establish the IMAP connection using imapclient.
        # imapclient handles SSL, timeout, and connection management.
        try:
            self._imap_connection = imapclient.IMAPClient(
                host=config.get("host"),
                port=config.get("port", 993),
                use_uid=True,
                ssl=True,
                timeout=timeout,
            )
        except Exception as e:
            _logger.error(f"[IMAPer] connection failed: {e}")
            self._imap_connection = None
            return

        # Log in and confirm the connection is established.
        try:
            self._imap_connection.login(config.get("username"), config.get("password"))
        except imapclient.Error as e:
            _logger.error(f"[IMAPer] login failed: {e}")
            self._imap_connection = None
            return

        _logger.info(f"[IMAPer] connected to {config.get('host')}:{config.get('port')}")

        # Populate the mailbox list after a successful login.
        # The tree refresh triggers _after_update_imap_tree which creates empty
        # envelope buffers for all mailboxes in the tree. Envelope content is
        # fetched lazily on first read via the BEFORE_READ_BUFFER hook.
        await self._refresh_imap_mailbox_list()

    async def _disconnect_imap(self) -> None:
        """Close the current IMAP connection if one exists.

        Logs out and sets _imap_connection to None.
        """
        # Bail out early if there is no active connection.
        if self._imap_connection is not None:
            try:
                self._imap_connection.logout()
            except Exception:
                # Swallow logout errors  --  the connection may already be dead.
                pass
            # Clear the reference to the connection.
            self._imap_connection = None
            _logger.info("[IMAPer] disconnected")

    def _list_imap_mailboxes(self) -> dict[str, str | list[list[str]]] | None:
        """List all IMAP mailbox names in the account.

        Returns a dict with the server's hierarchy "delimiter" and a "mailboxes"
        list. Each entry is a path list of segments, e.g. [["INBOX"], ["INBOX", "Work"]].
        The delimiter is preserved so the original server name can be reconstructed
        when needed.
        Raises RuntimeError if there is no active IMAP connection.
        Returns None if the server rejected the LIST command.
        """
        # Guard  --  reject if there is no active IMAP connection.
        if self._imap_connection is None:
            raise RuntimeError("no active IMAP connection")

        # List all folders recursively  --  imapclient returns already-parsed tuples.
        try:
            folders = self._imap_connection.list_folders()
        except imapclient.Error as e:
            # Server rejected the LIST command  --  log and return None.
            _logger.error(f"[IMAPer] LIST command error: {e}")
            return None

        # Format the result into the (status, entries) tuple that _parse_mailbox_list expects.
        # Each entry is formatted as: (flags, delimiter, name) like imaplib returns.
        if not folders:
            return None
        entries = [folder for folder in folders]
        return _parse_mailbox_list(("OK", entries))

    async def _refresh_imap_mailbox_list(self) -> None:
        """Refresh the email:tree buffer with the current list of mailboxes.

        Overwrites email:tree with a JSON list of path lists (e.g. [["INBOX"], ["INBOX", "Work"]]).
        Updates email:status with the delimiter, count, and a refreshed timestamp.
        """
        # Bail out if there is no active IMAP connection.
        if self._imap_connection is None:
            raise RuntimeError("no active IMAP connection")

        # Fetch the current mailbox list from the server.
        mailboxes = self._list_imap_mailboxes()
        if mailboxes is None:
            return

        # Build a nested tree from the flat path lists and write it to the box buffer.
        tree = _build_mailbox_tree(mailboxes.get("mailboxes", []))
        await self.create_buffer(
            _IMAPER_TREE_BUFFER,
            text=json.dumps(tree, ensure_ascii=False, indent=2),
            overwrite=True
        )

        # Update email:status with the current folder count, delimiter, and refresh time.
        status = {
            "mailbox_count": len(mailboxes.get("mailboxes", [])),
            "delimiter": mailboxes.get("delimiter", "/"),
            "last_refreshed_at": time.time(),
        }
        await self.create_buffer(_IMAPER_STATUS_BUFFER, text=json.dumps(status, ensure_ascii=False), overwrite=True)

    async def _refresh_imap_box(self, mailbox_name: str) -> None:
        """Reconcile UID buffers for a mailbox with the server and refresh the summary.

        Fetches the current UID list from the server, diffs it against the
        existing email:box/<mailbox>/* buffers, creates empty buffers for new UIDs,
        and drops buffers for UIDs no longer on the server.
        """
        # Bail out if there is no active IMAP connection.
        if self._imap_connection is None:
            raise RuntimeError("no active IMAP connection")

        # Read the filter buffer for this mailbox if it exists.
        # The filter holds {criteria, limit, offset} for the search.
        filter_buf = self.get_buffer(f"{_IMAPER_MAILBOX_SCOPE}/{mailbox_name}/envelopes.filter")
        filter_dict: dict | None = None
        if filter_buf and filter_buf.get_text().strip():
            try:
                filter_dict = json.loads(filter_buf.get_text())
            except json.JSONDecodeError:
                filter_dict = None

        # Fetch the current UID list from the server.
        result = _list_mailbox_emails(self._imap_connection, mailbox_name, filter=filter_dict)
        if result is None:
            return
        server_uids = set(result.get("uids", []))

        # Compute the prefix for this mailbox's UID buffers.
        mailbox_prefix = _IMAPER_MAILBOX_SCOPE + f"/{mailbox_name}/"

        # Read the full buffer list from the status buffer  --  no truncation.
        status_buf = self.get_buffer("system:status:buffers")
        current_uid_buffers: set[str] = set()
        if status_buf:
            try:
                records = json.loads(status_buf.get_text()) if status_buf.get_text().strip() else []
            except json.JSONDecodeError:
                records = []
            current_uid_buffers = {
                r["name"] for r in records
                if r["name"].startswith(mailbox_prefix) and r["name"][len(mailbox_prefix):].isdigit()
            }

        # Diff against desired UID buffers for this mailbox.
        desired_uid_buffers = {f"{mailbox_prefix}{uid}" for uid in server_uids}
        to_create = sorted(desired_uid_buffers - current_uid_buffers)
        to_drop = sorted(current_uid_buffers - desired_uid_buffers)

        # Create empty buffers for new UIDs  --  these are lazy-loaded on first read.
        for buf_name in to_create:
            await self.create_buffer(buf_name, text="")

        # Drop buffers for UIDs that no longer exist on the server.
        for buf_name in to_drop:
            await self.drop_buffer(buf_name)

        # Write the summary buffer with the full email list.
        buffer_name = _mailbox_to_envelopes_buffer_name(mailbox_name)
        await self.create_buffer(buffer_name, text=json.dumps(result, indent=2, ensure_ascii=False), overwrite=True)
