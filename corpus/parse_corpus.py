import os
import re
import argparse
from collections import defaultdict
from tqdm import tqdm

tokens_map = defaultdict(int)
functions_map = defaultdict(tuple)

MAX_PATH_NUM = 500


def parse_methods(f):
    line = f.readline()
    pbar.update(1)
    class_name = ""
    name = ""
    content = ""
    while line != "end_methods\n":
        if line.startswith("class:"):
            class_name = line[6:].rstrip("\n")
            line = f.readline()
            pbar.update(1)
        elif line.startswith("name:"):
            name = line[5:].strip("\n")
            line = f.readline()
            pbar.update(1)
        elif line == "content:\n":
            line = f.readline()
            pbar.update(1)
            while line != "end_content\n":
                content += line
                line = f.readline()
                pbar.update(1)
            key = class_name + ":" + name
            if not key in functions_map:
                functions_map[key] = [content, None]
                functions_map[key][1] = len(functions_map) - 1
            name = content = ""
            line = f.readline()
            pbar.update(1)
        else:
            pass


def parse_paths(f, sample_f):
    line = f.readline()
    pbar.update(1)
    sample = ""
    class_name = ""
    while line != "end_sample\n":
        if line.startswith("id:") or line.startswith("method:") or line.startswith("label:") or line.startswith("projectName:"):
            sample += line
            line = f.readline()
            pbar.update(1)
        elif line.startswith("class:"):
            sample += line
            class_name = line[6:].rstrip("\n")
            # print(class_name)
            line = f.readline()
            pbar.update(1)
        elif line.startswith("paths:"):
            sample += line
            line = f.readline()
            pbar.update(1)
            paths_num = 0
            while line != "end_sample\n":
                if paths_num > MAX_PATH_NUM:  # control paths num
                    line = f.readline()
                    pbar.update(1)
                    continue
                elements = line.split("\t")
                elements[-1] = elements[-1].rstrip("\n")
                for idx, element in enumerate(elements):
                    sample += "("
                    words = purify(element)
                    tokens = [x for x in words.split(" ") if x != ""]
                    if len(tokens) == 0:
                        tokens = ["null"]
                    if idx % 2 == 1:  # edge
                        flag = True
                        for token in tokens:
                            key = class_name + ":" + token
                            if key in functions_map:
                                sample = sample + "FUNC:" + str(functions_map[key][1]) + "\t"
                                flag = False
                                break

                        if flag:
                            for token in tokens:
                                n_token = split_token(token)
                                token_splits = n_token.split("|")
                                for token_split in token_splits:
                                    if token_split not in tokens_map:
                                        tokens_map[token_split] = len(tokens_map)
                                        sample = sample + str(len(tokens_map) - 1) + "|"
                                    else:
                                        sample = sample + str(tokens_map[token_split]) + "|"

                                sample = sample.rstrip("|")
                                sample += "\t"

                    else:  # node
                        for token in tokens:
                            n_token = split_token(token)
                            token_splits = n_token.split("|")
                            for token_split in token_splits:
                                if token_split not in tokens_map:
                                    tokens_map[token_split] = len(tokens_map)
                                    sample = sample + str(len(tokens_map) - 1) + "|"
                                else:
                                    sample = sample + str(tokens_map[token_split]) + "|"

                            sample = sample.rstrip("|")
                            sample += "\t"

                    sample = sample.rstrip("\t")
                    sample += ")"

                sample += "\n"
                paths_num += 1
                line = f.readline()
                pbar.update(1)
    sample += "\n"
    sample_f.write(sample)


def purify(element):
    word = re.sub(r"[^a-zA-Z0-9]+", " ", element)
    return word


def split_token(token):
    p = re.compile(r"([a-z]|\d)([A-Z])")
    p2 = re.compile(r"(\d)([a-z])")
    p3 = re.compile(r"([a-z])(\d)")
    sub = re.sub(p, r"\1|\2", token).lower()
    sub = re.sub(p2, r"\1|\2", sub)
    sub = re.sub(p3, r"\1|\2", sub)
    sub = re.sub(r"_", "|", sub)
    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path of the corpus.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="Output direction.")
    args = parser.parse_args()
    corpus = args.file
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sample_f = open(os.path.join(output_dir, "samples.txt"), "a", encoding="utf-8")

    with open(corpus, "rb") as f:
        # count number of lines for tqdm
        count = 0
        for count, _ in enumerate(f):
            pass
        count += 1

        # count number of samples
        n_samples = 0
        offset = -1
        f.seek(offset, os.SEEK_END)
        x = str(f.read(13), encoding="utf-8")
        while x != "begin_sample\n":
            offset -= 1
            f.seek(offset, os.SEEK_END)
            x = str(f.read(13), encoding="utf-8")
        n_samples = int(str(f.readline()[3:-1], encoding="utf-8")) + 1
        sample_f.write("n_samples:{}\n".format(n_samples))
        sample_f.flush()

    with open(corpus, "r", encoding="utf-8") as f:
        with tqdm(total=count) as pbar:
            pbar.set_description("Parsing")
            line = f.readline()
            pbar.update(1)
            while line:
                if line.startswith("methods:"):
                    parse_methods(f)
                elif line.startswith("begin_sample"):
                    parse_paths(f, sample_f)
                line = f.readline()
                pbar.update(1)
    sample_f.close()

    # save tokens
    with open(os.path.join(output_dir, "tokens.txt"), "w", encoding="utf-8") as tokens_file:
        ordered = sorted(tokens_map.items(), key=lambda x: x[1], reverse=False)
        for item in ordered:
            tokens_file.write(str(item[1]) + "\t" + item[0] + "\n")

    # save functions
    with open(os.path.join(output_dir, "functions.txt"), "w", encoding="utf-8") as f_file:
        for key, value in functions_map.items():
            id = str(value[1])
            content = value[0]
            class_name, name = key.split(":")
            f_file.write("id:" + id + "\n")
            f_file.write("classname:" + class_name + "\n")
            f_file.write("name:" + name + "\n")
            f_file.write("content:\n" + content + "end_content\n\n")
