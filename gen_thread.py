import threading, queue
import llama_cpp as llama
import random, torch, re

class LSSIJob:
    def __init__(self, bot, prompt=None, post_id=None, parent_id=None):
        self.bot = bot
        self.prompt = prompt
        self.post_id = post_id
        self.parent_id = parent_id
        self.complete_lock = threading.Lock()
        self.complete_lock.acquire()
        self.output_body = ""
        self.output_title = ""

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if self.bot.cfg["model"].endswith("gguf"):
            self.model = llama.Llama(self.bot.cfg["model"], use_mmap=True, use_mlock=True, n_ctx=1024, n_batch=1024,
                                     n_threads=6, n_threads_batch=12)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bot.cfg["model"])
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(self.bot.cfg["model"])
            self.model.eval()

    def gen(self, prompt=None):
        import warnings
        random.seed()
        if prompt is None:
            prompt = self.prompt
        self._load_model()
        self.bot.log.debug(f"Prompt:\n{prompt}\n\n")
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
        temp = random.uniform(float(self.bot.temprange[0]), float(self.bot.temprange[1]))

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
                                del self.model
                                return ""
                            out = self.model(prompt=prompt, temperature=float(temp), max_tokens=1024, stop=["<|"], seed=random.randint(0, 1000))["choices"][0]["text"]
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
            for b in self.bot.badwords:
                if b in out:
                    del self.model
                    return ""
            #txt = clean(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
            txt = out
            while "\\n" in txt:
                txt = txt.replace("\\n", " ")
            if txt and not self.bot.is_toxic(txt):
                del self.model
                return txt
        del self.model
        return ""

    def gen_post(self):
        title = ""
        sub = random.choice(self.bot.subreplace)
        selfsub = random.randint(0, 100) < 70
        for _ in range(3):
            if selfsub:
                raw = self.gen(f"<|soss r/{sub}|><|sot|>")
            else:
                raw = self.gen(f"<|sols r/{sub}|><|sot|>")
            c = raw.splitlines()[0][:200].strip()
            c = re.sub(r"[<>|].*?$", "", c)
            if c and c != "Untitled 🤔":
                title = c
                break
        # If still empty, seed the model with "Vile Asslips"
        if not title:
            seed = "Vile Asslips"
            fb = self.gen(seed).splitlines()[0][:200].strip()
            title = fb or seed

        # generate body by using title as the prompt
        body = ""
        if selfsub:
            for _ in range(3):
                candidate = self.gen(f"<|soss r/{sub}|><|sot|>{title}<|eot|><|sost|>").strip()
                # ensure it didn’t just echo the title
                if candidate and candidate.lower() != title.lower():
                    body = candidate
                    break
        if not body:
            body = " "  # Lemmy requires non‑empty

        self.output_body = body
        self.output_title = title

    def run(self):
        if self.prompt is None:
            self.gen_post()
            self.bot.post(title=self.output_title, body=self.output_body)
        else:
            self.output_body = self.gen(prompt=self.prompt)
            self.bot.comment(self.post_id, self.output_body, self.parent_id)
        self.complete_lock.release()

class GenThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="GenThread")
        self.genq = queue.Queue()

    def run(self):
        while True:
            try:
                job = self.genq.get()
                job.run()
            except Exception as e:
                print(repr(e))
