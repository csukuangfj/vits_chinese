#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import Dict, List

import onnxruntime
import soundfile
import torch


def display(sess):
    for i in sess.get_inputs():
        print(i)

    print("-" * 10)
    for o in sess.get_outputs():
        print(o)


class OnnxModel:
    def __init__(
        self,
        model: str,
    ):
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.model = onnxruntime.InferenceSession(
            model,
            sess_options=self.session_opts,
        )
        display(self.model)

        meta = self.model.get_modelmeta().custom_metadata_map
        self.add_blank = int(meta["add_blank"])
        self.sample_rate = int(meta["sample_rate"])
        print(meta)

    def __call__(
        self,
        x: torch.Tensor,
        sid: int,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A int64 tensor of shape (L,)
        """
        x = x.unsqueeze(0)
        x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
        noise_scale = torch.tensor([1], dtype=torch.float32)
        length_scale = torch.tensor([1], dtype=torch.float32)
        noise_scale_w = torch.tensor([1], dtype=torch.float32)
        sid = torch.tensor([sid], dtype=torch.int64)

        y = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_length.numpy(),
                self.model.get_inputs()[2].name: noise_scale.numpy(),
                self.model.get_inputs()[3].name: length_scale.numpy(),
                self.model.get_inputs()[4].name: noise_scale_w.numpy(),
                self.model.get_inputs()[5].name: sid.numpy(),
            },
        )[0]
        return torch.from_numpy(y).squeeze()


def read_lexicon() -> Dict[str, List[str]]:
    ans = dict()
    with open("./aishell3/lexicon.txt", encoding="utf-8") as f:
        for line in f:
            w_p = line.split()
            w = w_p[0]
            p = w_p[1:]
            ans[w] = p
    return ans


def read_tokens() -> Dict[str, int]:
    ans = dict()
    with open("./aishell3/tokens.txt", encoding="utf-8") as f:
        for line in f:
            t_i = line.strip().split()
            if len(t_i) == 1:
                token = " "
                idx = t_i[0]
            else:
                assert len(t_i) == 2, (t_i, line)
                token = t_i[0]
                idx = t_i[1]
            ans[token] = int(idx)
    return ans


def convert_lexicon(lexicon, tokens):
    for w in lexicon:
        phones = lexicon[w]
        try:
            p = [tokens[i] for i in phones]
            lexicon[w] = p
        except Exception:
            #  print("skip", w)
            continue


"""
skip rapprochement
skip croissants
skip aix-en-provence
skip provence
skip croissant
skip denouement
skip hola
skip blanc
"""


def get_text(text, lexicon, tokens):
    text = list(text)
    ans = []
    ans.append(tokens["sil"])
    for i in range(len(text)):
        w = text[i]
        if w == " ":
            ans.append(tokens["sil"])
            continue

        suffix = []

        if w in lexicon:
            ans.extend(lexicon[w])
            if suffix:
                ans.extend(suffix)

            continue
    ans.append(tokens["sil"])
    ans.append(tokens["eos"])
    return ans


def generate(model, text, lexicon, tokens, sid):
    x = get_text(
        text,
        lexicon,
        tokens,
    )
    if model.add_blank:
        x2 = [0] * (2 * len(x) + 1)
        x2[1::2] = x
        x = x2

    x = torch.tensor(x, dtype=torch.int64)

    y = model(x, sid=sid)

    return y


def main():
    model = OnnxModel("./aishell3/vits-aishell3.onnx")

    lexicon = read_lexicon()
    tokens = read_tokens()
    convert_lexicon(lexicon, tokens)

    # 6, 9, 10, 13, 16, 17, 18, 21, 27, 30, 32, 33, 34, 35, 36, 37, 40, 41, 42
    # 43, 44, 45, 46, 49
    text = "百川东到海 何时复西归 少壮不努力 老大徒伤悲"
    for sid in range(50):
        y = generate(model, text, lexicon, tokens, sid=sid)
        soundfile.write(f"test-{sid}.wav", y.numpy(), model.sample_rate)


if __name__ == "__main__":
    main()
