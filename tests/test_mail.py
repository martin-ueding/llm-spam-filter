from email.message import EmailMessage

from imap_tools.message import MailMessage

from llm_spam_organizer.mail import Mail


def build(*, text: str | None = None, html: str | None = None, attachment: bool = False) -> Mail:
    message = EmailMessage()
    message["From"] = "Sender <sender@example.com>"
    message["To"] = "me@example.com"
    message["Subject"] = "Free money"
    message["Date"] = "Sat, 15 Aug 2026 12:00:00 +0200"
    message["List-Unsubscribe"] = "<https://example.com/unsub>"
    if text is not None:
        message.set_content(text)
    if html is not None:
        if text is None:
            message.set_content("")
        message.add_alternative(html, subtype="html")
    if attachment:
        message.add_attachment(
            b"%PDF-1.4 fake", maintype="application", subtype="pdf", filename="invoice.pdf"
        )
    return Mail.from_message(MailMessage.from_bytes(message.as_bytes()), "INBOX")


def test_plain_text_body_is_kept():
    mail = build(text="Hello\n\n\n\nworld   \n")
    assert mail.body == "Hello\n\nworld"


def test_html_is_converted_when_there_is_no_text_part():
    mail = build(html="<html><body><p>Buy <b>now</b></p></body></html>")
    assert "Buy" in mail.body
    assert "<b>" not in mail.body


def test_attachments_appear_as_names_only():
    mail = build(text="see attached", attachment=True)
    assert mail.attachments == ["invoice.pdf"]
    rendered = mail.render(1000)
    assert "Attachments: invoice.pdf" in rendered
    assert "%PDF" not in rendered


def test_render_includes_headers_and_truncates():
    mail = build(text="x" * 500)
    rendered = mail.render(100)
    assert "From: Sender <sender@example.com>" in rendered
    assert "Subject: Free money" in rendered
    assert "List-Unsubscribe: <https://example.com/unsub>" in rendered
    assert "[truncated]" in rendered
    assert len(rendered) < 500


def test_round_trip_through_dict():
    mail = build(text="hello", attachment=True)
    assert Mail.from_dict(mail.to_dict()) == mail
