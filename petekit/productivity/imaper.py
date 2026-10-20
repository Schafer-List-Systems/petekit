from __future__ import annotations

from typing import Any

import imapclient
import json
import logging
import time

from email.header import decode_header

from peteos.oap.agentic_object import AgenticObject

from petekit.text.buffer_manager import Buffer, BufferManager, HookPoint


_logger = logging.getLogger("imaper")

_IMAPER_SCOPE = "imap:"
_IMAPER_DOC_BUFFER = "doc:imap"
_IMAPER_CONFIG_BUFFER = _IMAPER_SCOPE + "config"
_IMAPER_STATUS_BUFFER = _IMAPER_SCOPE + "status"
_IMAPER_TREE_BUFFER = _IMAPER_SCOPE + "tree"
_IMAPER_MAILBOX_SCOPE = _IMAPER_SCOPE + "box"
_IMAPER_DOC = f"""\
## IMAPer

The IMAPer manages one IMAP account for receiving emails via a buffer-based API.

### Buffers

- **{_IMAPER_CONFIG_BUFFER}** — holds IMAP credentials, subscribed mailboxes, and server settings
- **{_IMAPER_STATUS_BUFFER}** — holds runtime status such as IMAP connection state, folder count, and last refresh time
- **{_IMAPER_TREE_BUFFER}** — holds the mailbox tree as nested JSON (e.g. `{{"INBOX": {{"Sent": {{}}, "Work": {{}}}}}}`)
- **{_IMAPER_MAILBOX_SCOPE}/INBOX/envelopes** — holds the list of emails in INBOX with metadata (sender, recipient, subject, date, flags, uid)
- **{_IMAPER_MAILBOX_SCOPE}/INBOX/Sent/envelopes** — same for a subfolder; path segments use "/" as hierarchy separator

### Configuration

To configure the IMAP client, write to `imap:config` with a JSON object containing:

```json
{{
  "host": "imap.example.com",
  "port": 993,
  "username": "your-username",
  "password": "your-password",
  "timeout": 15,
  "subscribed_mailboxes": ["INBOX", "INBOX/Sent", "INBOX/Work"]
}}
```

All fields are optional. The `timeout` field controls connection timeout in seconds (defaults to 15).

The `subscribed_mailboxes` field is a list of mailbox paths to subscribe to. The IMAPer fetches the email list for each subscribed mailbox and writes it to the corresponding `imap:box/<path>/envelopes` buffer. Mailboxes not in this list are not monitored.

The IMAPer automatically connects when the config is written and reconnects on any credential change. It tries SSL first; if that fails, it falls back to a plain IMAP4 connection.

### Refreshing the Mailbox Tree

The `imap:tree` buffer auto-refreshes from the server on every read. The agent does not need to manage a trigger buffer — reading the tree is sufficient to keep it current.

### Refreshing a Single Mailbox Email List

The `imap:box/<mailbox>/envelopes` buffer auto-refreshes from the server on every read. The agent does not need to manage a trigger buffer — reading the envelope buffer is sufficient to keep it current.

### Subscribing to Mailboxes

Add or remove paths from the `subscribed_mailboxes` list in `imap:config` to control which mailboxes are monitored. Each subscribed mailbox gets its own `imap:box/<path>/envelopes` buffer containing a list of emails with the following metadata per entry:

```json
{{
  "mailbox": "INBOX",
  "uids": [1234, 1235],
  "emails": [
    {{
      "uid": 1234,
      "sender": "Alice <alice@example.com>",
      "recipient": "bob@example.com",
      "subject": "Meeting tomorrow",
      "date": "19 Oct 2026 09:15",
      "flags": ["seen", "answered"]
    }}
  ]
}}
```

The `flags` field lists the IMAP flags: `seen` (read), `answered`, `flagged`, `deleted`, `draft`.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Util                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #

def _get_imap_config(config_buffer: Buffer | None) -> dict:
    """Read and parse the imap:config buffer contents as JSON.

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
    # Reject non-OK responses — the command failed on the server side.
    if not raw_response or raw_response[0] != 'OK':
        _logger.error(f"[IMAPer] mailbox list response not OK: {raw_response[0] if raw_response else 'empty'}")
        return None

    # Pull the entries list from the IMAP response — skip on malformed response.
    entries = raw_response[1] if len(raw_response) > 1 else []

    # Pull the delimiter from the first entry — imapclient returns it as bytes at index 1.
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

    # Extract each mailbox name — entry[2] is the full path as a str already.
    # Split it by the delimiter to get the individual path segments.
    names = []
    for entry in entries:
        # Skip malformed entries — need a 3-tuple of (flags, delimiter, name).
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

    Each path list like ["INBOX", "Work", "Vorträge"] becomes a nested dict chain.
    The resulting tree uses empty dicts as leaf placeholders.

    Example:
        [["INBOX"], ["INBOX", "Sent"], ["INBOX", "Work", "Vorträge"]]
        becomes
        {"INBOX": {"Sent": {}, "Work": {"Vorträge": {}}}}.
    """
    # Start with an empty root — the trunk of the mailbox tree.
    root: dict = {}

    # Walk each path and hang it off the root or an existing branch.
    for path in paths:
        # Begin at the root for every path — descend from here.
        node = root

        # Walk each segment of the current path and build out the tree.
        for segment in path:
            # Create a new branch if this segment has not been seen yet.
            if segment not in node:
                node[segment] = {}

            # Step into the segment for the next iteration.
            node = node[segment]

    # The fully built tree is returned — empty dicts mark leaf nodes.
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

    # Select the mailbox — imapclient returns a clean dict, not raw bytes.
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
) -> dict | None:
    """Return the list of all emails in a mailbox with ENVELOPE and FLAGS metadata.

    Uses imapclient's select_folder, search, and fetch methods which return
    already-parsed Python types. Returns None on any failure (logged).

    Returns:
        {
            "mailbox": str,
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

    # Select the mailbox — imapclient handles state management.
    try:
        imap_connection.select_folder(mailbox_name)
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] SELECT failed for {mailbox_name}: {e}")
        return None

    # Search for all emails — imapclient returns a clean list of message IDs.
    try:
        message_ids = imap_connection.search(["ALL"])
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] SEARCH failed for {mailbox_name}: {e}")
        return None

    # Handle the empty mailbox case — return an empty list cleanly.
    if not message_ids:
        return {"mailbox": mailbox_name, "uids": [], "emails": []}

    # Fetch ENVELOPE and FLAGS for all message IDs in one batch.
    # imapclient returns: {id: {'ENVELOPE': Envelope, 'FLAGS': (...)}}
    try:
        result = imap_connection.fetch(message_ids, ["ENVELOPE", "FLAGS"])
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] FETCH failed for {mailbox_name}: {e}")
        return None

    # Parse each email from the fetched result.
    # imapclient returns already-parsed Envelope and FLAGS objects.
    emails: list = []
    for uid, fields in result.items():
        # Extract the ENVELOPE data — imapclient returns byte keys.
        envelope = fields.get(b'ENVELOPE')
        if envelope:
            # Format the sender address from the first Address in the from_ tuple.
            sender = ""
            if envelope.from_:
                addr = envelope.from_[0]
                sender = _format_imap_address(addr)

            # Format the first recipient from the to tuple.
            recipient = ""
            if envelope.to:
                addr = envelope.to[0]
                recipient = _format_imap_address(addr)

            # Extract subject and date — imapclient returns subject as bytes.
            # Decode RFC 2047 MIME encoded words (e.g. =?UTF-8?Q?...?=) to proper unicode.
            raw_subject = envelope.subject
            if isinstance(raw_subject, bytes):
                raw_subject = raw_subject.decode("utf-8")
            if raw_subject:
                parts = decode_header(raw_subject)
                subject = ''.join(
                    (p[0].decode(p[1]) if p[1] else p[0].decode('utf-8', errors='replace')) if isinstance(p[0], bytes) else str(p[0])
                    for p in parts
                )
            else:
                subject = ""
            date = str(envelope.date) if envelope.date else ""
        else:
            sender = ""
            recipient = ""
            subject = ""
            date = ""

        # Extract the FLAGS — imapclient uses byte keys and returns a tuple of byte flag names.
        # Decode each flag byte to a readable string.
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

    # Return the complete mailbox email list.
    return {
        "mailbox": mailbox_name,
        "uids": message_ids,
        "emails": emails,
    }


def _format_imap_address(address: imapclient.response_types.Address) -> str:
    """Format an imapclient Address object as a readable email address.

    Address(mailbox=u'alice', host=u'example.com', name=u'Alice') → "Alice <alice@example.com>"
    """
    # Reject a None address — nothing to format.
    if not address:
        return ""

    # Decode bytes fields to strings — Address fields may be bytes or str.
    def _decode(val):
        if isinstance(val, bytes):
            try:
                return val.decode("utf-8")
            except Exception:
                pass
        return str(val) if val else ""

    name = _decode(address.name)
    # Decode RFC 2047 MIME encoded display names (e.g. =?UTF-8?Q?=C3=9Cber?= Smith)
    if name.startswith("=?"):
        try:
            parts = decode_header(name)
            name = ''.join(
                (p[0].decode(p[1]) if p[1] else p[0].decode('utf-8', errors='replace')) if isinstance(p[0], bytes) else str(p[0])
                for p in parts
            )
        except Exception:
            pass
    mailbox = _decode(address.mailbox)
    host = _decode(address.host)

    # Build the full address string only when both mailbox and host are present.
    if mailbox and host:
        address_str = f"{mailbox}@{host}"
        # Prefix with the display name when one exists.
        if name:
            return f'"{name}" <{address_str}>'
        return address_str

    # Fall back to the bare name when the address is incomplete.
    return name


def _mailbox_to_envelopes_buffer_name(mailbox_name: str) -> str:
    """Convert a mailbox path to its corresponding imap:box envelope buffer name.

    INBOX         → imap:box/INBOX/envelopes
    INBOX/Sent    → imap:box/INBOX/Sent/envelopes
    """
    return _IMAPER_MAILBOX_SCOPE + f"/{mailbox_name}/envelopes"


def _fetch_email_by_uid(imap_connection: imapclient.IMAPClient, mailbox_name: str, uid: int) -> dict | None:
    """Fetch a single email by UID from the server.

    Returns a dict with uid, sender, recipient, subject, date, and flags.
    Returns None on any failure (logged).
    """
    # Reject if there is no active IMAP connection.
    if imap_connection is None:
        raise RuntimeError("no active IMAP connection")

    # Select the mailbox — required before fetching message data.
    try:
        imap_connection.select_folder(mailbox_name)
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] SELECT failed for {mailbox_name}: {e}")
        return None

    # Fetch the ENVELOPE and FLAGS for the single UID.
    try:
        result = imap_connection.fetch([uid], [b'ENVELOPE', b'FLAGS'])
    except imapclient.Error as e:
        _logger.error(f"[IMAPer] FETCH failed for UID {uid} in {mailbox_name}: {e}")
        return None

    # Bail out if the UID was not returned in the response.
    if not result or uid not in result:
        return None

    # Extract the envelope and flags from the response.
    fields = result[uid]
    envelope = fields.get(b'ENVELOPE')
    if not envelope:
        return None

    # Format the sender and recipient from the Address objects.
    sender = _format_imap_address(envelope.from_[0]) if envelope.from_ else ""
    recipient = _format_imap_address(envelope.to[0]) if envelope.to else ""

    # Decode the subject — imapclient may return bytes.
    raw_subject = envelope.subject
    if isinstance(raw_subject, bytes):
        raw_subject = raw_subject.decode("utf-8")
    if raw_subject:
        parts = decode_header(raw_subject)
        subject = ''.join(
            (p[0].decode(p[1]) if isinstance(p[0], bytes) and p[1] else str(p[0]))
            for p in parts
        )
    else:
        subject = ""

    # Extract the date and decode the FLAGS tuple to readable strings.
    date = str(envelope.date) if envelope.date else ""
    raw_flags = fields.get(b'FLAGS', [])
    flags = [f.decode("utf-8") if isinstance(f, bytes) else str(f) for f in raw_flags]

    return {
        "uid": uid,
        "sender": sender,
        "recipient": recipient,
        "subject": subject,
        "date": date,
        "flags": flags,
    }


def _flatten_tree(tree: dict) -> list[list[str]]:
    """Flatten a nested mailbox tree into a list of all leaf paths.

    Traverses the tree depth-first. Each leaf (an empty dict {}) becomes one
    path list — the list of keys from the root to that leaf.

    Example:
        {"INBOX": {"Sent": {}, "Work": {}}}
        → [["INBOX", "Sent"], ["INBOX", "Work"]]

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

            # Every key marks a path — if it has children we recorded the deeper
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
    """Register the validation and update hooks on the imap:config buffer.

    Called once when imap:config is created. Registers before_update_imap_config
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
    """Validate that the imap:config buffer contains valid JSON.

    Returns None to allow the update, or an error string to block it.
    """
    # Reject non-JSON config — read directly from the incoming buffer.
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
    """Called after imap:config is created or updated. Triggered on first config setup.

    On creation old_config_buffer is None — this signals initial setup rather than a
    later config change.
    """
    # Parse both configs so the IMAPer can diff and react to changes.
    old_config = _get_imap_config(old_config_buffer)
    new_config = _get_imap_config(new_config_buffer)

    # React to credential changes — disconnect old connection if host/port/user/pass changed.
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
    else:
        # No credential change — react to subscription changes only.
        pass

    # Diff the subscribed_mailboxes list and handle adds and removes.
    old_subscribed = set(old_config.get("subscribed_mailboxes", []))
    new_subscribed = set(new_config.get("subscribed_mailboxes", []))
    added = new_subscribed - old_subscribed
    removed = old_subscribed - new_subscribed

    # Handle removed subscriptions — drop their envelopes buffer.
    for mailbox in removed:
        # Derive the envelope buffer name from the mailbox path.
        envelopes_buf_name = _mailbox_to_envelopes_buffer_name(mailbox)

        # Drop the envelopes data buffer by name.
        await imaper.drop_buffer(envelopes_buf_name)

    # Handle added subscriptions — refresh their envelope buffer.
    # Only process adds when there was no credential change (otherwise _connect_imap handled it).
    if not _imap_connection_changed(old_config, new_config):
        for mailbox in added:
            # Refresh the envelopes buffer for the new subscription.
            if imaper._imap_connection is not None:
                await imaper._refresh_imap_box(mailbox)


# -------------------------------------------------------------------------------------------------------------------- #
# Buffer hooks                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


async def _before_read_envelopes(
    imaper: IMAPer,
    buffer: Buffer,
    args: dict[str, Any],
) -> bool | str | None:
    """Called before an envelopes buffer is read.

    Fires on any imap:box/<mailbox>/envelopes buffer. Extracts the mailbox name
    from the buffer name and triggers a refresh of the envelopes summary before
    the read proceeds. This keeps the email list current on every access.
    """
    # Extract the mailbox name from the buffer name.
    # The buffer name is: _IMAPER_MAILBOX_SCOPE / mailbox_name / envelopes
    # parts = ["imap:box", "<mailbox>", ..., "envelopes"]
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

    Only fires on imap:box/<mailbox>/<uid> buffers (pattern-matched).
    If the buffer already has content, proceeds without modification.
    If empty, fetches from the server and populates before allowing the read.
    """
    # Skip if the buffer already has content.
    if buffer.get_text():
        return True

    # Extract the UID from the buffer name — the last path segment is the UID.
    parts = buffer.name.split("/")
    uid = int(parts[-1])

    # Derive the mailbox name — the path between the mailbox scope and the UID.
    mailbox_name = "/".join(parts[1:-1])

    # Fetch the email from the server.
    email = _fetch_email_by_uid(imaper._imap_connection, mailbox_name, uid)
    if email is None:
        return f"[IMAPer] failed to fetch UID {uid} from {mailbox_name}"

    # Populate the buffer via set_text so hooks and references stay valid.
    buffer.set_text(json.dumps(email, indent=2))

    return True


async def _after_update_imap_tree(
    imaper: IMAPer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after imap:tree is updated.

    Fires whenever the tree buffer is written (via create_buffer with overwrite or write_buffer).
    Use this to drive any downstream state that depends on the current mailbox tree.
    On first creation old_buffer is None — this is initial setup, nothing to diff.
    """
    # Parse both trees from their buffer text. An empty or absent old buffer
    # means initial setup — there is nothing to diff against.
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

        # Seed the tree buffer with an empty list and register the update hook directly.
        self._create_buffer(_IMAPER_TREE_BUFFER, text="[]")
        tree_buffer = self.get_buffer(_IMAPER_TREE_BUFFER)
        if tree_buffer is None:
            raise RuntimeError(f"failed to create tree buffer '{_IMAPER_TREE_BUFFER}'")
        tree_buffer.register_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_update_imap_tree",
            _after_update_imap_tree,
        )
        tree_buffer.register_hook(
            HookPoint.BEFORE_READ_BUFFER,
            "refresh_tree_on_read",
            lambda *_: self._refresh_imap_mailbox_list(),
        )

        # Reject any attempt to drop the status buffer — it belongs to the IMAPer.
        status_buffer = self.get_buffer(_IMAPER_STATUS_BUFFER)
        if status_buffer is not None:
            status_buffer.register_hook(
                HookPoint.BEFORE_DROP_BUFFER,
                "imap_status_no_drop",
                lambda bm, buf, args: "imap:status cannot be dropped",
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

        # Register the lazy-fetch hook for UID buffers — fires before first read of each UID.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_READ_BUFFER,
            "lazy_fetch_imap_email",
            r"imap:box/[^/]+/\d+$",
            _before_read_imap_email,
            apply_to_existing=True,
        )

        # Register the envelopes before-read hook — stub for future per-mailbox logic.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_READ_BUFFER,
            "before_read_envelopes",
            r"imap:box/.+/envelopes$",
            _before_read_envelopes,
            apply_to_existing=True,
        )

    async def _update_imap_connection(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
    ) -> dict:
        """Read the existing imap:config, overlay the four credentials, write back if changed.

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
            await self.create_buffer(_IMAPER_CONFIG_BUFFER, text=json.dumps(merged), overwrite=True)

        return merged

    async def _connect_imap(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
    ) -> None:
        """Establish a new IMAP connection with the given credentials.

        Reads the current imap:config buffer, overlays the four credential arguments,
        writes the merged result back to imap:config as authoritative state.
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
        await self._refresh_imap_mailbox_list()

        # Refresh envelopes for all subscribed mailboxes.
        subscribed = config.get("subscribed_mailboxes", [])
        for mailbox in subscribed:
            # Fetch and cache the email list for this mailbox.
            await self._refresh_imap_box(mailbox)

    async def _disconnect_imap(self) -> None:
        """Close the current IMAP connection if one exists.

        Logs out and sets _imap_connection to None.
        """
        # Bail out early if there is no active connection.
        if self._imap_connection is not None:
            try:
                self._imap_connection.logout()
            except Exception:
                # Swallow logout errors — the connection may already be dead.
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
        # Guard — reject if there is no active IMAP connection.
        if self._imap_connection is None:
            raise RuntimeError("no active IMAP connection")

        # List all folders recursively — imapclient returns already-parsed tuples.
        try:
            folders = self._imap_connection.list_folders()
        except imapclient.Error as e:
            # Server rejected the LIST command — log and return None.
            _logger.error(f"[IMAPer] LIST command error: {e}")
            return None

        # Format the result into the (status, entries) tuple that _parse_mailbox_list expects.
        # Each entry is formatted as: (flags, delimiter, name) like imaplib returns.
        if not folders:
            return None
        entries = [folder for folder in folders]
        return _parse_mailbox_list(("OK", entries))

    async def _refresh_imap_mailbox_list(self) -> None:
        """Refresh the imap:tree buffer with the current list of mailboxes.

        Overwrites imap:tree with a JSON list of path lists (e.g. [["INBOX"], ["INBOX", "Work"]]).
        Updates imap:status with the delimiter, count, and a refreshed timestamp.
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
        await self.create_buffer(_IMAPER_TREE_BUFFER, text=json.dumps(tree), overwrite=True)

        # Update imap:status with the current folder count, delimiter, and refresh time.
        status = {
            "mailbox_count": len(mailboxes.get("mailboxes", [])),
            "delimiter": mailboxes.get("delimiter", "/"),
            "last_refreshed_at": time.time(),
        }
        await self.create_buffer(_IMAPER_STATUS_BUFFER, text=json.dumps(status), overwrite=True)

    async def _refresh_imap_box(self, mailbox_name: str) -> None:
        """Reconcile UID buffers for a mailbox with the server and refresh the summary.

        Fetches the current UID list from the server, diffs it against the
        existing imap:box/<mailbox>/* buffers, creates empty buffers for new UIDs,
        and drops buffers for UIDs no longer on the server.
        """
        # Bail out if there is no active IMAP connection.
        if self._imap_connection is None:
            raise RuntimeError("no active IMAP connection")

        # Fetch the current UID list from the server.
        result = _list_mailbox_emails(self._imap_connection, mailbox_name)
        if result is None:
            return
        server_uids = set(result.get("uids", []))

        # Compute the prefix for this mailbox's UID buffers.
        mailbox_prefix = _IMAPER_MAILBOX_SCOPE + f"/{mailbox_name}/"

        # Read the full buffer list from the status buffer — no truncation.
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

        # Create empty buffers for new UIDs — these are lazy-loaded on first read.
        for buf_name in to_create:
            await self.create_buffer(buf_name, text="")

        # Drop buffers for UIDs that no longer exist on the server.
        for buf_name in to_drop:
            await self.drop_buffer(buf_name)

        # Write the summary buffer with the full email list.
        buffer_name = _mailbox_to_envelopes_buffer_name(mailbox_name)
        await self.create_buffer(buffer_name, text=json.dumps(result, indent=2))
