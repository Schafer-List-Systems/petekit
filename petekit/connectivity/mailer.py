from __future__ import annotations

import email.message
import logging
import mimetypes
import smtplib
import ssl
from typing import Any


from peteos.oap.agentic_object import AgenticObject


from petekit.text.buffer_manager import Buffer, BufferManager, HookPoint


_logger = logging.getLogger("mailer")


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_MAILER_DOC_BUFFER = "doc:mailer"
_MAILER_DRAFT_SCOPE = "email:box/OUTBOX"
_MAILER_CONFIG_BUFFER = "email:config"

_MAILER_DOC = f"""\
## Mailer

Send emails by drafting them in a folder inside the `email:box/OUTBOX` mailbox.
Create one subfolder per draft. Each draft holds structured buffers that describe
the email.

### Draft Folder Structure

```
{_MAILER_DRAFT_SCOPE}/<draft-name>/
  header                    --  JSON with to, subject, cc, bcc, status, reviewed
  body                      --  plain text body
  html                      --  optional HTML body (mutually exclusive with body)
  attachments/meeting.ics   --  first attachment (any filename)
  attachments/notes.md      --  second attachment
  ...
```

### header Buffer

```json
{{
  "to": "<recipient@example.com>",
  "subject": "Hello",
  "cc": "<cc@example.com>",
  "bcc": "<bcc@example.com>",
  "status": "draft",
  "attachments": ["meeting.ics", "notes.md"]
}}
```

- `to`         --  required; the recipient email address
- `subject`    --  required; the email subject line
- `cc`         --  optional; comma-separated CC recipients
- `bcc`        --  optional; comma-separated BCC recipients
- `status`     --  write `send` to dispatch the email
- `attachments` --  list of attachment filenames; MIME type is guessed from each filename
- `reviewed`   --  internal; tracks which parts have been read; do not write manually

Setting `status` to `send` transmits the email and sets `status` to `sent`.
Drafts remain in the OUTBOX folder after sending.

### SMTP Configuration

The `{_MAILER_CONFIG_BUFFER}` buffer holds the SMTP credentials under a top-level `smtp` key:

```json
{{
  "smtp": {{
    "host": "smtp.example.com",
    "port": 587,
    "username": "me@example.com",
    "password": "secret",
    "use_ssl": false,
    "use_tls": true,
    "sender": "Me <me@example.com>"
  }}
}}
```

- `port`    --  465 for implicit SSL (SMTP_SSL), 587 for STARTTLS, 25 for plain
- `use_ssl` --  optional; defaults to true when port is 465
- `use_tls` --  optional; defaults to true when port is 587

### Sending

Write `send` to the header buffer's `status` field to trigger delivery.
Do NOT send without reviewing every part first.

You must read every part of the draft — header, body (or html), and each
attachment — before the send is allowed to proceed. Every modification to
a part requires another review of that part. This prevents you from
accidentally sending emails with mistakes or errors. Review the parts
thoroughly! Sending an email can NOT be undone!
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #


def _get_smtp_config(config_buffer: Buffer) -> dict | str | None:
    """Read the smtp sub-dict from the email:config buffer.

    The config JSON has a top-level "smtp" key containing:
      host       --  server hostname
      port       --  port number (465 for implicit SSL, 587 for STARTTLS, 25 plain)
      username   --  login username
      password   --  login password
      use_ssl    --  optional, default true for port 465, false otherwise
      use_tls    --  optional, default true for port 587, false otherwise

    Returns the smtp sub-dict on success, None if the smtp key is absent,
    or an error string if the config cannot be parsed.
    """
    # Parse the top-level config and return the smtp sub-section.
    import json
    try:
        config = json.loads(config_buffer.get_text())
    except json.JSONDecodeError as e:
        return f"{_MAILER_CONFIG_BUFFER} is not valid JSON: {e}"

    # Pull out the smtp sub-dict; bail out if it is absent.
    smtp = config.get("smtp")
    if not smtp:
        return f"{_MAILER_CONFIG_BUFFER} has no 'smtp' section"

    # Infer use_ssl and use_tls from the port when not explicitly set.
    port = smtp.get("port", 0)
    if smtp.get("use_ssl") is None:
        smtp["use_ssl"] = (port == 465)
    if smtp.get("use_tls") is None:
        smtp["use_tls"] = (port == 587)

    return smtp


def _build_email_message(
    mailer: Mailer,
    draft_name: str,
    smtp_config: dict,
) -> email.message.EmailMessage | str | None:
    """Read the header, body, and html buffers from a draft folder and compose a message.

    The draft folder is at `{_MAILER_DRAFT_SCOPE}/{draft_name}/` and must contain:
    - `header`   --  JSON with to, subject, cc, bcc, status
    - `body`     --  plain text body (optional if html is present)
    - `html`     --  HTML body (optional if body is present)

    The `sender` address is read from the smtp_config dict.  Returns the composed
    EmailMessage, or an error string if the header is absent or the message
    cannot be constructed.
    """
    # Locate the header buffer for this draft.
    header_buffer = mailer.get_buffer(f"{_MAILER_DRAFT_SCOPE}/{draft_name}/header")
    if header_buffer is None:
        return f"draft '{draft_name}' has no header buffer"

    # Parse the header JSON for the address and subject fields.
    import json
    try:
        header = json.loads(header_buffer.get_text() or "{}")
    except json.JSONDecodeError as e:
        return f"header buffer of '{draft_name}' is not valid JSON: {e}"

    # Extract the required recipient and subject; bail out if they are absent.
    to_addr = header.get("to", "")
    subject = header.get("subject", "")
    if not to_addr:
        return f"draft '{draft_name}' has no 'to' address in header"

    # Read the sender address from the smtp config.
    sender = smtp_config.get("sender", "")
    if not sender:
        return f"{_MAILER_CONFIG_BUFFER} smtp section has no 'sender' address"

    # Collect the optional CC and BCC addresses as a list.
    cc_list = [s.strip() for s in header.get("cc", "").split(",") if s.strip()]
    bcc_list = [s.strip() for s in header.get("bcc", "").split(",") if s.strip()]

    # Try to read the body and HTML buffers (mutually exclusive).
    body_text = ""
    html_text = ""
    body_buffer = mailer.get_buffer(f"{_MAILER_DRAFT_SCOPE}/{draft_name}/body")
    if body_buffer is not None:
        body_text = body_buffer.get_text() or ""
    html_buffer = mailer.get_buffer(f"{_MAILER_DRAFT_SCOPE}/{draft_name}/html")
    if html_buffer is not None:
        html_text = html_buffer.get_text() or ""

    # Build the EmailMessage with the appropriate body variant.
    message = email.message.EmailMessage()
    message["From"] = sender
    message["To"] = to_addr
    if cc_list:
        message["Cc"] = ", ".join(cc_list)
    if subject:
        message["Subject"] = subject

    if html_text and not body_text:
        # HTML only  --  set it as the message body.
        message.set_content(html_text, subtype="html", charset="utf-8")
    elif body_text:
        # Plain text  --  set it as the message body.
        message.set_content(body_text, subtype="plain", charset="utf-8")
    else:
        # Neither body nor HTML present  --  leave the message empty.
        pass

    # Add every attachment from the draft; guess the MIME type from the
    # attachment name and read the buffer content as raw bytes.
    for attachment_name in header.get("attachments", []):
        # Locate the attachment buffer and read its content; skip if absent.
        attach_buffer = mailer.get_buffer(
            f"{_MAILER_DRAFT_SCOPE}/{draft_name}/attachments/{attachment_name}"
        )
        if attach_buffer is None:
            continue

        # Extract the text content; skip if the buffer is empty.
        attach_bytes = attach_buffer.get_text()
        if not attach_bytes:
            continue

        # Encode the string content to bytes if needed for binary detection.
        if isinstance(attach_bytes, str):
            attach_bytes = attach_bytes.encode("utf-8")
        if not attach_bytes:
            continue

        # Guess the MIME type from the filename; default to octet-stream.
        mime_type = mimetypes.guess_type(attachment_name)
        if mime_type:
            maintype, subtype = mime_type
        else:
            maintype, subtype = "application", "octet-stream"

        # Add the attachment to the message; email.message handles
        # Content-Transfer-Encoding and multipart conversion automatically.
        message.add_attachment(
            attach_bytes,
            filename=attachment_name,
            maintype=maintype or "application",
            subtype=subtype or "octet-stream",
        )

    return message


def _send_email(
    smtp_cfg: dict,
    sender: str,
    recipients: list[str],
    message: email.message.EmailMessage,
) -> str | None:
    """Blocking SMTP send.  Returns None on success, error string on failure."""
    # Catch any SMTP error so we can log it and return cleanly.
    try:
        # For port 465: use SMTP_SSL for implicit TLS on connect.
        if smtp_cfg["use_ssl"]:
            # Create a secure default SSL context for the TLS handshake.
            context = ssl.create_default_context()

            # Establish an implicit-TLS SMTP connection and send.
            with smtplib.SMTP_SSL(smtp_cfg["host"], smtp_cfg["port"], context=context) as conn:
                # Upgrade to STARTTLS if configured and not already implicit.
                if smtp_cfg["use_tls"] and not smtp_cfg["use_ssl"]:
                    conn.starttls(context=context)

                # Authenticate with the server credentials.
                conn.login(smtp_cfg["username"], smtp_cfg["password"])

                # Hand off the composed message for delivery.
                conn.send_message(message, sender, recipients)

        # For other ports: use plain SMTP with optional STARTTLS upgrade.
        else:
            # Connect in plain text mode then upgrade if requested.
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as conn:
                # Upgrade to STARTTLS on ports like 587.
                if smtp_cfg["use_tls"]:
                    conn.starttls()

                # Authenticate with the server credentials.
                conn.login(smtp_cfg["username"], smtp_cfg["password"])

                # Hand off the composed message for delivery.
                conn.send_message(message, sender, recipients)

        # Report success so the caller knows the message was accepted.
        return None

    # Surface the error so the caller can act on the failure.
    except Exception as e:
        _logger.error(f"[Mailer] SMTP send failed: {e}")
        return str(e)


def _list_draft_parts(
    mailer: Mailer,
    draft_scope: str,
) -> list[str]:
    """Return the buffer names of all parts that belong to the given draft.

    Reads the header buffer from the draft scope, then derives body, html,
    and all attachment paths from it.  Returns a list of full buffer names, e.g.:
      [email:box/OUTBOX/my-draft/body, email:box/OUTBOX/my-draft/html,
       email:box/OUTBOX/my-draft/attachments/1]
    """
    # Read the header to get the attachments list; bail out if absent.
    import json
    header_buffer = mailer.get_buffer(f"{draft_scope}/header")
    if header_buffer is None:
        return []
    try:
        header = json.loads(header_buffer.get_text() or "{}")
    except json.JSONDecodeError:
        return []

    # Always include body and html if they exist in the buffer store.
    part_names: list[str] = []
    body_buffer = mailer.get_buffer(f"{draft_scope}/body")
    if body_buffer is not None:
        part_names.append(f"{draft_scope}/body")
    html_buffer = mailer.get_buffer(f"{draft_scope}/html")
    if html_buffer is not None:
        part_names.append(f"{draft_scope}/html")

    # Enumerate every attachment index from the header and include it if
    # the buffer actually exists.
    for attachment_name in header.get("attachments", []):
        attachment_path = f"{draft_scope}/attachments/{attachment_name}"
        if mailer.get_buffer(attachment_path) is not None:
            part_names.append(attachment_path)

    return part_names


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


async def _before_update_header(
    mailer: Mailer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before the header buffer of a draft is updated.

    When the `status` field is set to `send`, this triggers delivery.
    Returns an error string on failure, None to allow the update.
    """
    # Parse the header JSON and check that the required fields are present.
    import json
    try:
        header = json.loads(new_buffer.get_text() or "{}")
    except json.JSONDecodeError as e:
        return f"header is not valid JSON: {e}"

    if not isinstance(header, dict):
        return "header must be a JSON object"

    if not isinstance(header.get("to"), str) or not header.get("to", "").strip():
        return "header must contain a non-empty 'to' field"

    if not isinstance(header.get("subject"), str):
        return "header must contain a 'subject' field"

    # Only enforce the review check on a fresh send transition  --
    # read the old header to determine the previous status.
    old_status = None
    if old_buffer is not None:
        try:
            old_header = json.loads(old_buffer.get_text() or "{}")
            if isinstance(old_header, dict):
                old_status = old_header.get("status")
        except json.JSONDecodeError:
            pass

    # When status transitions to send, verify that every draft part has been
    # reviewed before allowing the send to proceed.
    if old_status != "send" and header.get("status") == "send":
        # Derive the draft scope from the buffer name, e.g. email:box/OUTBOX/draft1.
        buffer_name = new_buffer.name
        draft_scope = buffer_name.removesuffix("/header")

        # Get the list of all parts that belong to this draft.
        draft_parts = _list_draft_parts(mailer, draft_scope)

        # Build the set of already-reviewed part names from the header.
        reviewed_list = header.get("reviewed", [])
        if not isinstance(reviewed_list, list):
            reviewed_list = []

        # Reject the send if any draft part has not yet been reviewed.
        unreviewed_parts = [
            part for part in draft_parts
            if part not in reviewed_list
        ]
        if unreviewed_parts:
            return (
                f"cannot send: the following parts have not been reviewed yet: "
                f"{', '.join(unreviewed_parts)}  --  review each part before sending"
            )

    # Optional fields are not checked.
    return None


async def _after_update_header(
    mailer: Mailer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> bool:
    """Called after the header buffer of a draft is updated.

    When the status transitions to send, this triggers the actual delivery.
    After sending, the status is changed to sent.
    """
    # Parse the new header to read the new status.
    import json
    try:
        header = json.loads(new_buffer.get_text() or "{}")
    except json.JSONDecodeError:
        return True

    # Reject the update if the header is not a JSON object.
    if not isinstance(header, dict):
        return True

    # Pull the new status from the freshly-parsed header.
    new_status = header.get("status")

    # Read the old status to detect a transition to send.
    old_status = None
    if old_buffer is not None:
        # Parse the old header to extract its status; ignore corrupt content.
        try:
            old_header = json.loads(old_buffer.get_text() or "{}")
            if isinstance(old_header, dict):
                old_status = old_header.get("status")
        except json.JSONDecodeError:
            pass

    # Only proceed when the status transitions to send.
    if old_status == "send" or new_status != "send":
        return True

    # Derive the draft name and scope from the buffer name.
    # The buffer name is e.g. email:box/OUTBOX/draft1/header.
    buffer_name = new_buffer.name
    draft_scope = buffer_name.removesuffix("/header")
    draft_name = draft_scope.rsplit("/", 1)[-1]

    # Load the SMTP config from email:config; abort if not present.
    config_buffer = mailer.get_buffer(_MAILER_CONFIG_BUFFER)
    if config_buffer is None:
        _logger.error("[Mailer] cannot send: email:config buffer not found")
        return True

    # Read the smtp sub-dict from the config; handle all error cases.
    smtp_cfg = _get_smtp_config(config_buffer)
    if isinstance(smtp_cfg, str):
        _logger.error(f"[Mailer] cannot send: {smtp_cfg}")
        return True
    if smtp_cfg is None:
        _logger.error("[Mailer] cannot send: email:config has no smtp section")
        return True

    # Build the email message from the draft; bail out if construction fails.
    message_result = _build_email_message(mailer, draft_name, smtp_cfg)
    if isinstance(message_result, str):
        _logger.error(f"[Mailer] cannot send: {message_result}")
        return True
    if message_result is None:
        _logger.error(f"[Mailer] cannot send: draft '{draft_name}' has no message content")
        return True

    # Send the message via SMTP; report the error if sending fails.
    sender = smtp_cfg.get("sender", "")
    recipients = [header.get("to", "")]
    send_error = _send_email(smtp_cfg, sender, recipients, message_result)
    if send_error is not None:
        _logger.error(f"[Mailer] send failed: {send_error}")
        return True

    # Confirm success and record that the draft was sent.
    _logger.info(f"[Mailer] email sent successfully for draft '{draft_name}'")

    # Update the status to sent in the header so the agent can see the result.
    header["status"] = "sent"
    write_result = await mailer.create_buffer(
        buffer_name,
        text=json.dumps(header, indent=2, ensure_ascii=False),
        overwrite=True,
    )
    if not write_result.get("ok"):
        _logger.error(
            f"[Mailer] failed to update status to sent in '{buffer_name}': "
            f"{write_result.get('error', 'unknown')}"
        )

    return True


async def _after_read_draft_part(
    mailer: Mailer,
    buffer: Buffer,
    args: dict[str, Any],
) -> bool:
    """Called after a draft part buffer is read.

    Marks the read buffer in the draft's header under the `reviewed` list so
    the agent can track which parts are already fresh.
    """
    # Derive the draft folder by stripping the part segment from the buffer name.
    # Derive the draft prefix by checking the last path segment.
    # body and html are one segment; attachments/N strips two segments.
    buffer_name = buffer.name
    parts = buffer_name.rsplit("/", 2)
    if parts[-1] in ("body", "html"):
        # body and html are single-segment leaves; strip one from the right.
        draft_prefix = buffer_name.rsplit("/", 1)[0]
    elif len(parts) >= 2 and parts[1] == "attachments":
        # attachments/N is two segments deep; strip two from the right.
        draft_prefix = buffer_name.rsplit("/", 2)[0]
    else:
        return True

    # Locate the header buffer for this draft; bail out if it does not exist.
    header_name = f"{draft_prefix}/header"
    header_buffer = mailer.get_buffer(header_name)
    if header_buffer is None:
        return True

    # Parse the header JSON; bail out if the content is corrupt.
    import json
    try:
        header = json.loads(header_buffer.get_text() or "{}")
    except json.JSONDecodeError:
        return True

    # Reject the update if the header is not a JSON object.
    if not isinstance(header, dict):
        return True

    # Initialise the reviewed list if it is absent, then append this buffer
    # only if it is not already recorded.
    reviewed_list = header.get("reviewed")
    if not isinstance(reviewed_list, list):
        reviewed_list = []
        header["reviewed"] = reviewed_list

    # Append this buffer to the reviewed list only if it is not already there.
    if buffer.name not in reviewed_list:
        reviewed_list.append(buffer.name)

    # Write the updated header so the agent sees the reviewed list grow.
    result = await mailer.create_buffer(
        header_name,
        text=json.dumps(header, indent=2, ensure_ascii=False),
        overwrite=True,
    )
    if not result.get("ok"):
        _logger.error(
            f"[Mailer] failed to update reviewed list in '{header_name}': "
            f"{result.get('error', 'unknown')}"
        )

    return True


async def _after_update_draft_part(
    mailer: Mailer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> bool:
    """Called after a draft part buffer is updated.

    Removes the updated buffer from the header's reviewed list so it must
    be read again before the email can be sent.
    """
    # Derive the draft folder from the updated buffer name.
    buffer_name = new_buffer.name
    rsplit_result = buffer_name.rsplit("/", 1)
    # Bail out if the buffer name has no slash  --  an impossible case.
    if len(rsplit_result) < 2:
        _logger.error(f"[Mailer] buffer name '{buffer_name}' has no slash  --  impossible")
        return True
    draft_prefix = rsplit_result[0]

    # Locate the header buffer for this draft; bail out if it does not exist.
    header_name = f"{draft_prefix}/header"
    header_buffer = mailer.get_buffer(header_name)
    if header_buffer is None:
        return True

    # Parse the header JSON; bail out if the content is corrupt.
    import json
    try:
        header = json.loads(header_buffer.get_text() or "{}")
    except json.JSONDecodeError:
        return True

    # Reject the update if the header is not a JSON object.
    if not isinstance(header, dict):
        return True

    # Remove the updated buffer from the reviewed list so it must be
    # re-read before sending is allowed again.
    reviewed_list = header.get("reviewed")
    if isinstance(reviewed_list, list) and buffer_name in reviewed_list:
        reviewed_list.remove(buffer_name)

    # Write the updated header so the agent sees the reviewed list shrink.
    write_result = await mailer.create_buffer(
        header_name,
        text=json.dumps(header, indent=2, ensure_ascii=False),
        overwrite=True,
    )
    if not write_result.get("ok"):
        _logger.error(
            f"[Mailer] failed to update reviewed list in '{header_name}': "
            f"{write_result.get('error', 'unknown')}"
        )

    return True


async def _after_create_attachment(
    mailer: Mailer,
    buffer: Buffer,
    args: dict[str, Any],
) -> bool:
    """Called after an attachment buffer is created in a draft folder.

    Adds the new attachment buffer name to the header's attachments list.
    """
    # Derive the draft folder from the new attachment buffer name.
    # The path is: email:box/OUTBOX/<draft>/attachments/<filename>
    # Split into 3 parts from the right: [draft_prefix, "attachments", name]
    buffer_name = buffer.name
    rsplit_result = buffer_name.rsplit("/", 2)
    if len(rsplit_result) < 3:
        _logger.error(f"[Mailer] buffer name '{buffer_name}' missing segments")
        return
    draft_prefix = rsplit_result[0]
    attachment_name = rsplit_result[2]

    # Locate the header buffer for this draft; bail out if it does not exist.
    header_name = f"{draft_prefix}/header"
    header_buffer = mailer.get_buffer(header_name)
    if header_buffer is None:
        _logger.error(f"[Mailer] header buffer '{header_name}' not found")
        return True

    # Parse the header JSON; bail out if the content is corrupt.
    import json
    try:
        header = json.loads(header_buffer.get_text() or "{}")
    except json.JSONDecodeError as e:
        _logger.error(f"[Mailer] header '{header_name}' is not valid JSON: {e}")
        return True

    # Reject the update if the header is not a JSON object.
    if not isinstance(header, dict):
        _logger.error(f"[Mailer] header '{header_name}' is not a JSON object")
        return True

    # Initialise the attachments list if it is absent, then append this
    # attachment only if it is not already recorded.
    attachments_list = header.get("attachments")
    if not isinstance(attachments_list, list):
        attachments_list = []
        header["attachments"] = attachments_list

    if attachment_name not in attachments_list:
        attachments_list.append(attachment_name)

    # Write the updated header so the agent sees the attachments list grow.
    write_result = await mailer.create_buffer(
        header_name,
        text=json.dumps(header, indent=2, ensure_ascii=False),
        overwrite=True,
    )
    if not write_result.get("ok"):
        _logger.error(
            f"[Mailer] failed to update attachments list in '{header_name}': "
            f"{write_result.get('error', 'unknown')}"
        )

    return True


async def _after_drop_attachment(
    mailer: Mailer,
    buffer: Buffer,
    args: dict[str, Any],
) -> bool:
    """Called after an attachment buffer is dropped from a draft folder.

    Removes the dropped attachment from the header's attachments list.
    """
    # Derive the draft folder from the updated buffer name.
    # The path is: email:box/OUTBOX/<draft>/attachments/<filename>
    # Split into 3 parts from the right: [draft_prefix, "attachments", name]
    buffer_name = buffer.name
    rsplit_result = buffer_name.rsplit("/", 2)
    if len(rsplit_result) < 3:
        _logger.error(f"[Mailer] buffer name '{buffer_name}' missing segments")
        return True
    draft_prefix = rsplit_result[0]
    attachment_name = rsplit_result[2]

    # Locate the header buffer for this draft; bail out if it does not exist.
    header_name = f"{draft_prefix}/header"
    header_buffer = mailer.get_buffer(header_name)
    if header_buffer is None:
        _logger.error(f"[Mailer] header buffer '{header_name}' not found")
        return True

    # Parse the header JSON; bail out if the content is corrupt.
    import json
    try:
        header = json.loads(header_buffer.get_text() or "{}")
    except json.JSONDecodeError as e:
        _logger.error(f"[Mailer] header '{header_name}' is not valid JSON: {e}")
        return True

    # Reject the update if the header is not a JSON object.
    if not isinstance(header, dict):
        _logger.error(f"[Mailer] header '{header_name}' is not a JSON object")
        return True

    # Remove the attachment name from the attachments list if it is present.
    attachments_list = header.get("attachments")
    if isinstance(attachments_list, list) and attachment_name in attachments_list:
        attachments_list.remove(attachment_name)

    # Write the updated header so the agent sees the attachments list shrink.
    write_result = await mailer.create_buffer(
        header_name,
        text=json.dumps(header, indent=2, ensure_ascii=False),
        overwrite=True,
    )
    if not write_result.get("ok"):
        _logger.error(
            f"[Mailer] failed to update attachments list in '{header_name}': "
            f"{write_result.get('error', 'unknown')}"
        )

    return True


# -------------------------------------------------------------------------------------------------------------------- #
# Mailer                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


class Mailer(BufferManager, AgenticObject):
    """I can draft and send emails.
    `doc:mailer` holds details.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_MAILER_DOC_BUFFER, text=_MAILER_DOC)

        # Register the hook on all header buffers under the draft scope.
        # When status is set to send, delivery is triggered.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "mailer_send_trigger",
            rf"^{_MAILER_DRAFT_SCOPE}/.+/header$",
            _before_update_header,
            apply_to_existing=True,
        )
        # Also register on first creation  --  the CREATE signature takes 3 args
        # so a lambda bridges the gap by inserting None for old_buffer.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "mailer_send_trigger",
            rf"^{_MAILER_DRAFT_SCOPE}/.+/header$",
            lambda bm, new_buffer, args: _before_update_header(bm, None, new_buffer, args),
            apply_to_existing=True,
        )
        # Register the AFTER_UPDATE hook on header buffers to trigger send.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "mailer_send",
            rf"^{_MAILER_DRAFT_SCOPE}/.+/header$",
            _after_update_header,
            apply_to_existing=True,
        )

        # Register the AFTER_READ hook on body, html, and attachment buffers.
        # One pattern with alternation catches all three cases.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_READ_BUFFER,
            "mailer_read_draft_part",
            rf"^{_MAILER_DRAFT_SCOPE}/.+/(body|html|attachments/[^/]+)$",
            _after_read_draft_part,
            apply_to_existing=True,
        )
        # Register the AFTER_UPDATE hook on body, html, and attachment buffers.
        # When a part changes, it is removed from the reviewed list.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "mailer_update_draft_part",
            rf"^{_MAILER_DRAFT_SCOPE}/.+/(body|html|attachments/[^/]+)$",
            _after_update_draft_part,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "mailer_drop_draft_part",
            rf"^{_MAILER_DRAFT_SCOPE}/.+/(body|html|attachments/[^/]+)$",
            lambda bm, new_buffer, args: _after_update_draft_part(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register hooks on attachment buffers for create and drop events.
        attachment_pattern = rf"^{_MAILER_DRAFT_SCOPE}/.+/attachments/[^/]+$"
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "mailer_create_attachment",
            attachment_pattern,
            _after_create_attachment,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "mailer_drop_attachment",
            attachment_pattern,
            _after_drop_attachment,
            apply_to_existing=True,
        )


