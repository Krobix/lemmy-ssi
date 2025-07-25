import re
from typing import Any
import pythorhead
# ------------------------------------------------------------------ #
#  Helpers                                                         #
# ------------------------------------------------------------------ #
CLEAN_TOKENS = [
    "<|soss|>", "<|sot|>", "<|eot|>", "<|sost|>", "<|eost|>",
    "<|sols|>", "<|eols|>", "<|sor|>", "<|eor|>", "<|sol|>", "<|eol|>", "<|eoss|>",
]
_ZWJ_RE = re.compile(r"[\u200d\uFE0F]")

def clean(text: str) -> str:
    for tok in CLEAN_TOKENS:
        text = text.replace(tok, " ")
    return _ZWJ_RE.sub("", text).strip()

def split_title_body(raw: str) -> tuple[str, str]:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return "Untitled 🤔", ""
    title = re.sub(r"[<>|].*?$", "", lines[0][:200]).strip() or "Untitled 🤔"
    body = "\n".join(lines[1:]).strip()
    return title, body

def iter_post_views(raw: Any):
    if isinstance(raw, dict):
        yield from raw.get("posts", [])
    elif isinstance(raw, list):
        for item in raw:
            if "post" in item and "creator" in item:
                yield item
            else:
                yield {"post": item, "creator": {"name": item.get("name","")}}

def iter_comment_views(raw: Any):
    if isinstance(raw, dict):
        yield from raw.get("comments", [])
    elif isinstance(raw, list):
        for item in raw:
            if "comment" in item and "creator" in item:
                yield item
            else:
                yield {"comment": item, "creator": {"name": item.get("creator_name","")}}


def convert_post(title, text, sub, is_self=True, end_tag=False):
    out = ""

    if is_self:
        out += f"<|soss r/{sub}|>"
    else:
        out += f"<|sols r/{sub}|>"

    out += f"<|sot|>{title}"

    if is_self:
        out += f"<|sost|>{text}"
    if end_tag:
        if is_self:
            out += "<|eoss|>"
        else:
            out += "<|eols|>"
    return out

def convert_thread(post, replies, sub, is_self=True):
    out = convert_post(post["post"]["name"], post["post"]["body"], sub, end_tag=False)
    if out is None:
        return None

    parent_author = None
    parent_2_author = None
    for r in replies:
        if r['creator']["name"] == post['creator']["name"]:
            out += "<|soopr|>"
        elif r['creator']["name"] == parent_2_author:
            out += "<|soocr|>"
        else:
            out += f"<|sor u/{r['creator']["name"]}|>"
        out += r["comment"]['content']
        parent_2_author = parent_author
        parent_author = r['creator']["name"]

    #if is_self:
    #    out += "<|eoss|><|endoftext|>\n"
    #else:
    #    out += "<|eols|><|endoftext|>\n"
    # print("Thread converted")
    return out

def get_sort_type():
    try:
        from pythorhead.enums import SortType
    except ImportError:
        try:
            from pythorhead.const import SortType
        except ImportError:
            class _EnumPlaceholder(str):
                @property
                def value(self) -> str:
                    return str(self)

            class SortType:
                New = _EnumPlaceholder("New")

    return SortType