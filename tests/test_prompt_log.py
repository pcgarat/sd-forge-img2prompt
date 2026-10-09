from forge_img2prompt import prompt_log


def test_record_and_last_raw_roundtrip():
    prompt_log.clear()
    assert prompt_log.last_raw() == ""
    assert prompt_log.render_markdown().startswith("_Sin peticiones")
    prompt_log.record(
        transport="Ollama POST /api/chat",
        payload={"model": "qwen3-vl:4b", "messages": [{"role": "user", "content": "hi"}]},
        params={"num_predict": 320, "temperature": 0.2},
        images=[],
    )
    raw = prompt_log.last_raw()
    assert "Ollama POST /api/chat" in raw
    assert "qwen3-vl:4b" in raw
    assert "num_predict" in raw
    prompt_log.clear()
    assert prompt_log.last_raw() == ""


def test_history_keeps_newest_first_and_capped():
    prompt_log.clear()
    for i in range(prompt_log._MAX_ENTRIES + 3):
        prompt_log.record(
            transport=f"call-{i}",
            payload={"i": i},
            params=None,
            images=[],
        )
    entries = prompt_log.history()
    assert len(entries) == prompt_log._MAX_ENTRIES
    assert "call-12" in entries[0].text
    assert "call-3" in entries[-1].text
    rendered = prompt_log.render_markdown()
    assert "call-12" in rendered
    assert "---" in rendered


def test_redact_images_truncates_base64():
    long_b64 = "A" * 5000
    payload = {
        "messages": [
            {"role": "user", "content": "x", "images": [long_b64]},
        ]
    }
    redacted = prompt_log.redact_images(payload)
    stored = redacted["messages"][0]["images"][0]
    assert len(stored) < 5000
    assert "5000 b64 chars" in stored
    # Original untouched
    assert payload["messages"][0]["images"][0] == long_b64


def test_redact_data_urls_truncates_openai_payload():
    long_b64 = "B" * 5000
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{long_b64}"}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
    }
    redacted = prompt_log.redact_data_urls(payload)
    url = redacted["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) < 5000
    assert "5000 b64 chars" in url
    assert payload["messages"][0]["content"][0]["image_url"]["url"].endswith(long_b64)


def test_summarize_vl_messages_replaces_images():
    payload = {
        "role": "user",
        "content": [
            {"type": "image", "image": object()},
            {"type": "text", "text": "describe"},
        ],
    }
    summarized = prompt_log.summarize_vl_messages(payload)
    assert summarized["content"][0]["image"] != object()
    assert summarized["content"][1]["text"] == "describe"


def test_record_includes_image_summary_with_pil():
    from PIL import Image

    prompt_log.clear()
    prompt_log.record(
        transport="transformers generate",
        payload={"role": "user"},
        images=[Image.new("RGB", (32, 16))],
    )
    raw = prompt_log.last_raw()
    assert "32×16 RGB" in raw
