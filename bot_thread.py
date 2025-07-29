import threading, logging, random
from types import MappingProxyType
import torch
import time
from detoxify import Detoxify
from pythorhead import Lemmy
from util import *
from gen_thread import LSSIJob

SortType = get_sort_type()

# ------------------------------------------------------------------ #
#  Bot Thread                                                      #
# ------------------------------------------------------------------ #
class BotThread(threading.Thread):
    def __init__(self, bot_cfg: MappingProxyType, global_cfg: MappingProxyType, genq):
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

        self.genq = genq
        self.jobs = []

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

    def _add_job(self, prompt=None, post_id=None, parent_id=None):
        job = LSSIJob(bot=self, prompt=prompt, post_id=post_id, parent_id=parent_id)
        self.jobs.append(job)
        self.genq.put(job)
        if parent_id is not None:
            self.replied_to.append(parent_id)

    def is_toxic(self, txt: str, thresh: float = 0.9) -> bool:
        if not self.filter_toxic:
            return False
        try:
            return any(v > thresh for v in self.detox.predict(txt).values())
        except Exception:
            return False

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
        try:
            post = self.lemmy.post.get(post_id=c["comment"]["post_id"])["post_view"]
        except TypeError:
            self.log.debug("_org_thread: comment was None. continuing")
            return None, None
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
        time.sleep(5) # fix ratelimit problems
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
                if post is None: continue
                #self.log.info(post)
                p = convert_thread(post, replies, sub) + f"<|sor u/{self.cfg['username']}|>"
                parent_id = src["comment"]["id"]
            elif "post" in src:
                try:
                    p = convert_post(title=src["post"]["name"], text=src["post"]["body"], sub=sub) + f"<|sor u/{self.cfg["username"]}|>"
                except KeyError:
                    p = convert_post(title=src["post"]["name"], text="", sub=sub) + f"<|sor u/{self.cfg["username"]}|>"
                parent_id = src["post"]["id"]
            else:
                continue
            if attempts >= self.max_replies:
                break
            random.seed(parent_id*int.from_bytes(self.cfg["username"].encode("utf-8"), byteorder="big", signed=False))
            if random.randint(1, 100) < self.roll_needed:
                continue
            if "comment" in src:
                self._add_job(prompt=p, post_id=src["comment"]["post_id"], parent_id=src["comment"]["id"])
            else:
                self._add_job(prompt=p, post_id=src["post"]["id"], parent_id=src["post"]["id"])
            attempts += 1
            time.sleep(random.uniform(self.delay_min, self.delay_max))

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                now = time.time()
                # (1) Post new thread – immediate on startup if initial_post=True
                if self.initial_post or (now - self.last_post_at) >= self.freq_s:
                    # Try to generate a title up to 3 times
                    self.comm_this_period = 0

                    self._add_job()
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

                #check for complete jobs
                for i in range(len(self.jobs)):
                    job = self.jobs.pop(0)
                    if job.complete_lock.acquire(blocking=False):
                        if job.parent_id is not None:
                            body = job.output_body
                            if body=="":
                                continue
                            self._comment(job.post_id, body, parent_id=job.parent_id)
                        else:
                            self._post(job.output_title, job.output_body)
                    else:
                        self.jobs.append(job)

                time.sleep(300)
            except Exception as e:
                self.log.error(repr(e))