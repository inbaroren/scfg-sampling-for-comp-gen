import subprocess
import os, shutil
import argparse
from inspect import currentframe, getframeinfo


def do(commnad):
    result = subprocess.run(commnad, check=True, shell=True, universal_newlines=True)


def get_domain(domain, output, maxdepth, prunesize, memsize, k=""):
    """
    Augments synthetic data for domain, save it to {output}/{domain}_augmented.tsv
    :param domain: domain of the data
    :param output: path to output directory
    :param maxdepth: maximum depth of the generated data
    :param prunesize: how many partial examples remains during generation
    :param memsize: memory size for process
    :param k: developer key for genie toolkit
    :return: None
    """
    # output_dir = os.path.join(output, domain)
    output_dir = output
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
    # get the manifest, make sure annotation=bart only
    do(f"make experiment={domain} annotation=manual {domain}/manifest.tt")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)

    do(f"node --experimental_worker --max_old_space_size=15000 ../../tool/genie.js augment "
       f"-o {output_dir}/{domain}_augmented.tsv -l en-US --thingpedia {domain}/manifest.tt "
       f"--parameter-datasets {domain}/parameter-datasets.tsv --synthetic-expand-factor 1  "
       f"--quoted-paraphrasing-expand-factor 60 --no-quote-paraphrasing-expand-factor 20 "
       f"--quoted-fraction 0.0  --debug  {output_dir}/{domain}.tsv")
    frameinfo = getframeinfo(currentframe())
    print(frameinfo.filename, frameinfo.lineno)
    # copy the directory with the data
    # copytree(f"/genie-toolkit/starter/schemaorg/{domain}",
    #          output_dir)
    with open(f"{output_dir}/{domain}_augmented.tsv") as fp:
        lines = fp.readlines()
        print(f"There are {len(lines)} in {output_dir}/{domain}_augmented.tsv")


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
