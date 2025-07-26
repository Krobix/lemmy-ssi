import threading, logging, random
from types import MappingProxyType
import torch
import time
from detoxify import Detoxify
from pythorhead import Lemmy
import llama_cpp as llama
from util import *

SortType = get_sort_type()

# ------------------------------------------------------------------ #
#  Bot Thread                                                      #
# ------------------------------------------------------------------ #
class BotThread(threading.Thread):
    def __init__(self, bot_cfg: MappingProxyType, global_cfg: MappingProxyType, genlock):
        super().__init__(daemon=True, name=bot_cfg["name"])
        self.cfg = bot_cfg
        self.global_cfg = global_cfg
        self.log = logging.getLogger(bot_cfg["name"])

        # Lemmy login
        self.lemmy = Lemmy(global_cfg["instance"])
        self.lemmy.log_in(bot_cfg["username"], bot_cfg["password"])
        self.community_id = self.lemmy.discover_community(global_cfg["community"])

        self.subreplace = self.cfg["subreplace"].strip().split(",")
        self.nsfw = self.cfg["nsfw"]

        # Model + tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if bot_cfg["model"].endswith("gguf"):
            self.model = llama.Llama(bot_cfg["model"], use_mmap=True, use_mlock=True, n_ctx=1024, n_batch=1024, n_threads=6, n_threads_batch=12)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(bot_cfg["model"])
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(bot_cfg["model"])
            self.model.eval()

        # Toxicity
        self.filter_toxic = bool(global_cfg.get("toxicity_filter", True))
        self.detox = Detoxify("original", device="cpu")

        # Timing
        self.freq_s = float(bot_cfg.get("postfreq", 1)) * 3600
        self.initial_post = bool(global_cfg.get("initial_post", False))
        self.roll_needed = int(global_cfg.get("comment_roll", 70))
        self.max_replies = int(global_cfg.get("max_replies", 5))
        self.delay_min = float(global_cfg.get("reply_delay_min", 5))
        self.delay_max = float(global_cfg.get("reply_delay_max", 12))
        self.last_post_at = 0.0

        self.temprange = self.cfg["temprange"]
        self.temprange = self.temprange.strip().split(",")
        self.temprange[0] = float(self.temprange[0])
        self.temprange[1] = float(self.temprange[1])

        self.badwords = self.global_cfg["banned_words"].strip().split(",")
        self.replied_to = []
        self.comm_lim = 500
        self.comm_this_period = 0

        self.stop_event = threading.Event()
        self.genlock = genlock

    def _is_toxic(self, txt: str, thresh: float = 0.9) -> bool:
        if not self.filter_toxic:
            return False
        try:
            return any(v > thresh for v in self.detox.predict(txt).values())
        except Exception:
            return False

    def _gen(self, prompt: str) -> str:
        import warnings
        with self.genlock:
            self.log.debug(f"Prompt:\n{prompt}\n\n")
            if type(self.model) is not llama.Llama:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                ids = inputs.input_ids
                attn = inputs.get("attention_mask", None)
                # empty‑prompt fallback
                if not prompt.strip() or ids.size(1) == 0:
                    bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
                    ids = torch.tensor([[bos]], device=ids.device)
                    prompt_len = 1
                    attn = torch.ones_like(ids)
                else:
                    prompt_len = ids.size(1)
            else:
                prompt_len = len(self.model.tokenize(prompt.encode('utf-8')))
            temp = random.uniform(float(self.temprange[0]), float(self.temprange[1]))

            max_new = max(16, 1024 - prompt_len)
            for _ in range(4):
                with torch.no_grad():
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        if type(self.model) is llama.Llama:
                            try:
                                tokens = self.model.tokenize(prompt.encode("utf-8"))
                                tokens.reverse()
                                tokens = tokens[:1000]
                                tokens.reverse()
                                try:
                                    prompt = self.model.detokenize(tokens).decode("utf-8")
                                except UnicodeDecodeError:
                                    return ""
                                out = self.model(prompt=prompt, temperature=float(temp), max_tokens=1024, stop=["<|"])["choices"][0]["text"]
                                out = str(out)
                            except RuntimeError:
                                continue
                            #if out.endswith("<|"):
                            #    out = out[:len(out) - 2]
                            #if prompt.endswith("<|sot|>") and prompt.startswith("<|soss"):
                            #    logging.info("Title generated for text post, proceeding to generate text post body")
                            #    out = str(out) + "<|eot|><|sost|>"
                            #    out += str(self.model(prompt=out, temperature=float(temp), max_tokens=1024,
                            #                        stop=["<|"])["choices"][0]["text"])
                            #if not out.endswith("<|"):
                            #    out += "<|"
                        else:
                            out = self.model.generate(
                                ids,
                                attention_mask=attn,
                                max_new_tokens=max_new,
                                do_sample=True,
                                temperature=0.9,
                                top_p=0.9,
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                gen_ids = out[len(prompt):]
                for b in self.badwords:
                    if b in out:
                        return ""
                #txt = clean(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
                txt = out
                while "\\n" in txt:
                    txt = txt.replace("\\n", " ")
                if txt and not self._is_toxic(txt):
                    return txt
            return ""

    def _post(self, title: str, body: str) -> int | None:
        try:
            res = self.lemmy.post.create(self.community_id, title, body=body, nsfw=self.nsfw)
            pid = res["post_view"]["post"]["id"]
            self.log.info("Posted: %s", title)
            return pid
        except Exception:
            self.log.exception("post failed")
            return None

    def _comment(self, post_id: int, content: str, parent_id: int | None = None) -> None:
        try:
            self.lemmy.comment.create(post_id, content, parent_id=parent_id)
            self.log.info("Commented on %d", post_id)
            self.comm_this_period += 1
        except Exception:
            self.log.exception("comment failed")

    def _org_thread(self, c):
        post = self.lemmy.post.get(post_id=c["comment"]["post_id"])["post_view"]
        replies = [c]
        path = c["comment"]["path"].strip().split(".")
        path.pop(0)
        path.reverse()
        path = path[1:]
        for p in path:
            try:
                replies.append(self.lemmy.comment.get(int(p))["comment_view"])
            except:
                break
        replies.reverse()
        return post, replies

    def _already_replied(self, pv):
        post_id = pv["post"]["id"]
        if "comment" in pv:
            parent_id = pv["comment"]["id"]
            parid = parent_id
            max_depth = 10
        elif "post" in pv:
            parent_id = pv["post"]["id"]
            parid = None
            max_depth = 1
        else:
            return False

        if parent_id in self.replied_to:
            return True

        replies = self.lemmy.comment.list(post_id=post_id, parent_id=parid, max_depth=max_depth, limit=500)
        self.log.debug(f"Got {len(replies)} replies to pv")
        for r in replies:
            if "post_view" in r:
                r = r["post_view"]
            if "comment_view" in r:
                r = r["comment_view"]
            if r["creator"]["name"]==self.cfg["username"]:
                self.replied_to.append(parent_id)
                if len(self.replied_to) > 5000:
                    self.replied_to = []
                self.log.debug(f"Already replied")
                return True
        return False

    def _attempt_replies(self, sources: list[dict[str, Any]], sub) -> None:
        attempts = 0
        for src in sources:
            if src["creator"]["name"] == self.cfg["username"]:
                continue
            if "comment" in src:
                post, replies = self._org_thread(src)
                #self.log.info(post)
                p = convert_thread(post, replies, sub) + f"<|sor u/{src['creator']['name']}|>"
            elif "post" in src:
                p = convert_post(title=src["post"]["name"], text=src["post"]["body"], sub=sub) + f"<|sor u/{src['creator']['name']}|>"
            else:
                continue
            if attempts >= self.max_replies:
                break
            if random.randint(1, 100) < self.roll_needed:
                continue
            reply = ""
            for _ in range(3):
                candidate = self._gen(p).strip()
                if candidate:
                    reply = candidate
                    break
            if not reply:
                continue
            if "comment" in src:
                self._comment(src["comment"]["post_id"], reply, parent_id=src["comment"]["id"])
            else:
                self._comment(src["post"]["id"], reply, parent_id=src["post"]["id"])
            attempts += 1
            time.sleep(random.uniform(self.delay_min, self.delay_max))

    def run(self) -> None:
        while not self.stop_event.is_set():
            now = time.time()

            # (1) Post new thread – immediate on startup if initial_post=True
            if self.initial_post or (now - self.last_post_at) >= self.freq_s:
                # Try to generate a title up to 3 times
                self.comm_this_period = 0
                title = ""
                sub = random.choice(self.subreplace)
                for _ in range(3):
                    raw = self._gen(f"<|soss r/{sub}|><|sot|>")
                    c = raw.splitlines()[0][:200].strip()
                    c = re.sub(r"[<>|].*?$", "", c)
                    if c and c != "Untitled 🤔":
                        title = c
                        break
                # If still empty, seed the model with "Vile Asslips"
                if not title:
                    seed = "Vile Asslips"
                    fb = self._gen(seed).splitlines()[0][:200].strip()
                    title = fb or seed

                # generate body by using title as the prompt
                body = ""
                for _ in range(3):
                    candidate = self._gen(f"<|soss r/{sub}|><|sot|>{title}<|eot|><|sost|>").strip()
                    # ensure it didn’t just echo the title
                    if candidate and candidate.lower() != title.lower():
                        body = candidate
                        break
                if not body:
                    body = " "  # Lemmy requires non‑empty

                self._post(title, body)
                self.last_post_at = now
                self.initial_post = False  # only run once

            # 2) Scan & reply every loop
            feed = self.lemmy.post.list(
                page=1, limit=self.max_replies * 3,
                sort=SortType.New, community_id=self.community_id
            )
            posts = []

            if self.comm_this_period < self.comm_lim:
                for pv in iter_post_views(feed):
                    if self._already_replied(pv):
                        continue
                    posts.append(pv)
                sub = random.choice(self.subreplace)
                self._attempt_replies(posts, sub=sub)

                cfeed = self.lemmy.comment.list(
                    community_id=self.community_id,
                    sort=SortType.New, page=1, limit=self.max_replies * 3
                )
                comments = []
                for cv in iter_comment_views(cfeed):
                    if self._already_replied(cv):
                        continue
                    comments.append(cv)
                self._attempt_replies(comments, sub=sub)

            time.sleep(300)