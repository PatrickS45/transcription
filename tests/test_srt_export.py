from models import Subtitle, Word
from srt_export import format_timestamp, output_paths, render_srt, write_srt


def make_sub(text_lines, start, end):
    return Subtitle(words=[Word(" ".join(text_lines), start, end)],
                    lines=text_lines, start=start, end=end)


def test_timestamp_format():
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(83.5) == "00:01:23,500"
    assert format_timestamp(3661.042) == "01:01:01,042"
    assert format_timestamp(-1.0) == "00:00:00,000"


def test_render_srt_structure():
    subs = [
        make_sub(["Bonjour à tous", "et bienvenue"], 0.0, 2.5),
        make_sub(["Deuxième sous-titre"], 3.0, 5.0),
    ]
    content = render_srt(subs)
    blocks = content.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].splitlines() == [
        "1", "00:00:00,000 --> 00:00:02,500", "Bonjour à tous", "et bienvenue",
    ]
    assert blocks[1].splitlines()[0] == "2"


def test_write_utf8_no_bom(tmp_path):
    path = tmp_path / "out.srt"
    write_srt([make_sub(["Éléphant à l'œuvre"], 0, 1)], path, bom=False)
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "Éléphant à l'œuvre" in raw.decode("utf-8")


def test_write_utf8_with_bom(tmp_path):
    path = tmp_path / "out.srt"
    write_srt([make_sub(["Été"], 0, 1)], path, bom=True)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_output_paths_naming(tmp_path):
    paths = output_paths(tmp_path / "ma_video.mp4")
    assert paths["16-9"].name == "ma_video_16-9.srt"
    assert paths["9-16"].name == "ma_video_9-16.srt"
