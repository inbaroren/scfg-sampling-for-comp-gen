import subprocess
import os, shutil
import argparse
from inspect import currentframe, getframeinfo


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def do(commnad):
    result = subprocess.run(commnad, check=True, shell=True, universal_newlines=True)


def get_domain(domain, output, maxdepth, prunesize, memsize, k=""):
    """
    Generate synthetic data for domain, save it to {output}/{domain}_synthetic.tsv
    :param domain: domain of the data
    :param output: path to output directory
    :param maxdepth: maximum depth of the generated data
    :param prunesize: how many partial examples remains during generation
    :param memsize: memory size for process
    :param k: developer key for genie toolkit
    :return: None
    """
    output_dir = os.path.join(output, domain)
    os.makedirs(output_dir, exist_ok=True)

    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    # make sure the developer key is up to date
    # do("cd /genie-toolkit/starter/schemaorg")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    do(f"sed -i 's/memsize := 15000/memsize := {memsize}/g' Makefile")
    do(
        f"sed -i 's/developer_key =/developer_key = {k}/g' Makefile")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    do("sed -i 's/(genie) schemaorg-process-schema/(genie) schemaorg-process-schema --url https:\/\/raw.githubusercontent.com\/schemaorg\/schemaorg\/main\/data\/releases\/9.0\/schemaorg-current-http.jsonld/g' Makefile")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    # get the manifest, make sure annotation=bart only
    do(f"make experiment={domain} annotation=bart {domain}/manifest.tt")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    # create synthetic data, make sure the "augment" command has --parallelize 4 flag
    do(
        f"sed -i 's/maxdepth ?= 8/maxdepth ?= {maxdepth}/g' Makefile")
    do(
        f"sed -i 's/target_pruning_size ?= 500/target_pruning_size ?= {prunesize}/g' Makefile")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    do(
        "sed -i 's/--quoted-paraphrasing-expand-factor 60/--parallelize 4 --quoted-paraphrasing-expand-factor 80/g' Makefile")
    do(
        "sed -i 's/--synthetic-expand-factor 1 --quoted/--synthetic-expand-factor 8 --quoted/g' Makefile")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    do(f"make experiment={domain} datadir")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    # copy the directory with the data
    # copytree(f"/genie-toolkit/starter/schemaorg/{domain}",
    #          output_dir)
    with open(f"/genie-toolkit/starter/schemaorg/{domain}/synthetic.tsv") as fp:
        lines = fp.readlines()
        print(f"There are {len(lines)} in {domain}/synthetic.tsv")
    with open(f"/genie-toolkit/starter/schemaorg/{domain}/augmented.tsv") as fp:
        lines = fp.readlines()
        print(f"There are {len(lines)} in {domain}/augmented.tsv")
    do(f"cp -r /genie-toolkit/starter/schemaorg/{domain} {output_dir}")
    with open(f"{output_dir}/{domain}/synthetic.tsv") as fp:
        lines = fp.readlines()
        print(f"There are {len(lines)} in {output_dir}/{domain}/synthetic.tsv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True,
                        choices=['hotels', 'people', 'movies', 'music', 'books', 'restaurants'])
    parser.add_argument('--output-path', default="/opt/project", type=str)
    parser.add_argument('--maxdepth', default=10, type=int)
    parser.add_argument('--prunesize', default=1200, type=int)
    parser.add_argument('--memsize', default=15000, type=int)
    args = parser.parse_args()

    get_domain(args.domain, args.output_path, args.maxdepth, args.prunesize, args.memsize)
