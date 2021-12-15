import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import editdistance
from collections import defaultdict
import random
import heapq
from tqdm import tqdm
import pickle
import argparse

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


logger = logging.getLogger(__file__)


def creat_split_baseline(cs_df,
                         domain,
                         test_templs_prop,
                         dev_templs_prop,
                         validation_set_prop=0.05,
                         debug=False,
                         split_col='template_untyped_noops'):
    """
    split the annotated data compositionally
    """
    # Declare CS templates as compositional_test
    must_be_train_templates = {
        "now => ( schema_table ) filter param:property REL_OP VALUE LOG_OP param:property REL_OP VALUE => notify",
        "now => ( schema_table ) filter param:property REL_OP VALUE => notify",
        "now => [ param:property ] of ( ( schema_table ) filter param:property REL_OP VALUE ) => notify"}
    comp_templates = list(set(cs_df[split_col].unique()).difference(must_be_train_templates))
    if debug: print(len(comp_templates))
    comp_train_templs, comp_test_templs = train_test_split(comp_templates, test_size=test_templs_prop)
    comp_train_templs, comp_dev_templs = train_test_split(comp_train_templs, test_size=dev_templs_prop)

    train_cs_df = cs_df[
        cs_df[split_col].isin(
            comp_train_templs + list(must_be_train_templates))]
    comp_dev_cs_df = cs_df[
        cs_df[split_col].isin(
            comp_dev_templs)]
    comp_test_cs_df = cs_df[
        cs_df[split_col].isin(
            comp_test_templs)]

    train_cs_df, iid_test_cs_df = train_test_split(train_cs_df, test_size=validation_set_prop)
    train_cs_df, dev_cs_df = train_test_split(train_cs_df, test_size=validation_set_prop)

    # validate all the splits!!!
    comp_test_cs_df = validate_and_fix(comp_test_cs_df, train_cs_df[2], "comp test", "train", prog_col=2, debug=True)
    comp_dev_cs_df = validate_and_fix(comp_dev_cs_df, train_cs_df[2], "comp dev", "train", prog_col=2, debug=True)
    iid_test_cs_df = validate_and_fix(iid_test_cs_df, train_cs_df[2], "iid test", "train", prog_col=2, debug=True)
    dev_cs_df = validate_and_fix(dev_cs_df, train_cs_df[2], "dev", "train", prog_col=2, debug=True)

    print(train_cs_df.shape[0], comp_test_cs_df.shape[0], comp_dev_cs_df.shape[0], iid_test_cs_df.shape[0],
          dev_cs_df.shape[0])
    return train_cs_df, comp_test_cs_df, comp_dev_cs_df, iid_test_cs_df, dev_cs_df


def create_synth_splits_random_onefile(trainsize_list,
                                       synth_train_df,
                                       prog_col=2,
                                       num_splits=5,
                                       save_space=True,
                                       with_paraphrase=False):
    """
    create 5 uniform samples of each size s in trainsize_list
    """
    pull_size = synth_train_df.shape[0]
    print("Pull train size: ", pull_size)
    for s in trainsize_list:
        for i in range(num_splits):
            split_name = f"{s}_{i}_{with_paraphrase}"
            size = s
            if s > pull_size:
                print("WARNING: not enough examples. changed to max: ", pull_size)
                size = pull_size

            tmp_train, tmp_dev = train_test_split(synth_train_df.sample(n=size), test_size=0.05)
            tmp_dev = validate_and_fix(tmp_dev, tmp_train[prog_col], f"dev size {s}", f"train size {s}",
                                       prog_col=prog_col,
                                       debug=True)
            # merge
            tmp_train = tmp_train.assign(**{split_name: 1})
            tmp_dev = tmp_dev.assign(**{split_name: 2})
            tmp = pd.concat([tmp_train, tmp_dev], axis=0)
            synth_train_df = synth_train_df.merge(tmp[[split_name]], left_index=True, right_index=True, how='left')

            assert synth_train_df[split_name].value_counts()[1] == tmp_train.shape[0]
            assert synth_train_df[split_name].value_counts()[2] == tmp_dev.shape[0]
            assert synth_train_df.shape[0] == pull_size

    def is_chosen(x):
        for s in trainsize_list:
            for i in range(5):
                split_name = f"{s}_{i}_{with_paraphrase}"
                if x[split_name] == 1 or x[split_name] == 2:
                    return True
        return False

    if save_space:
        synth_train_df = synth_train_df[synth_train_df.apply(is_chosen, axis=1)]
    print(f"Train pull size: {synth_train_df.shape[0]}")
    return synth_train_df


def create_prog_split_skew_templates_freq(train_data_df,
                                          exp_list,
                                          training_size_list,
                                          validation_set_prop=0.05,
                                          temp_col='kbfree_untyped_noops',
                                          input_col=1,
                                          prog_col=2,
                                          num_splits=5,
                                          save_space=True,
                                          with_paraphrase: bool = False):
    """
    UAT sampling
    train_data_df: pd.DataFrame, pull of examples to sample from
    exp_list: float in [-1, 0]. the exponent (alpha) to use to skew the template distribution before sampling. -1 is closest to uniform, 0 is random
    training_size_list: List[int], list of sample sizes to sample
    validation_set_prop: float, proportion of the validation set
    temp_col: string, name of the template column in train_pull
    input_col: int, index of the question column in train_pull
    prog_col: int, index of the queries column in train_pull
    num_splits: int, how many training sets to sample from each size in training_size_list and each alpha in exp_list
    save_space: boolean, if true returns only examples that were sampled at least once
    with_paraphrase: boolean, indicates whether the input data includes paraphrased questions
    """
    # calculate frequncy of templates p(T=t), attach to all examples
    N = train_data_df.shape[0]
    freq = train_data_df[temp_col].value_counts(normalize=True).reset_index().rename(
        columns={temp_col: 'freq', 'index': temp_col})
    train_data_df = train_data_df.merge(freq, on=temp_col, how='left')

    # sample training sets by skewing p(T=t)
    for training_size in training_size_list:
        for exp in exp_list:
            for i in range(num_splits):
                split_name = f"{exp}_{training_size}_{i}_{with_paraphrase}".lower()
                # fix prob distribution
                weights = train_data_df['freq'].apply(lambda x: x ** exp)
                # sample
                sample = train_data_df.sample(n=training_size, weights=weights)
                tmp_train, tmp_dev = train_test_split(sample, test_size=validation_set_prop)
                # validate
                tmp_dev = validate_and_fix(tmp_dev, tmp_train[prog_col],
                                           f"dev alpha-{exp} training_size-{training_size} split number-{i}", f"train",
                                           prog_col=prog_col, debug=True)
                # merge
                tmp_train = tmp_train.assign(**{split_name: 1})
                tmp_dev = tmp_dev.assign(**{split_name: 2})
                tmp = pd.concat([tmp_train, tmp_dev], axis=0)
                train_data_df = train_data_df.merge(tmp[[split_name]], left_index=True, right_index=True, how='left')

                try:
                    assert len(train_data_df[split_name].value_counts()) == 2, "there are no examples in dev, try to increase the training size"
                    assert train_data_df[split_name].value_counts()[1] == tmp_train.shape[0]
                    assert train_data_df[split_name].value_counts()[2] == tmp_dev.shape[0]
                    assert train_data_df.shape[0] == N
                except KeyError:
                    print("there are no examples in dev, try to increase the training size\n", train_data_df[split_name].value_counts())

    def is_chosen(x):
        for s in training_size_list:
            for exp in exp_list:
                for i in range(num_splits):
                    split_name = f"{exp}_{s}_{i}_{with_paraphrase}"
                    if x[split_name] == 1 or x[split_name] == 2:
                        return True
        return False

    if save_space:
        train_data_df = train_data_df[train_data_df.apply(is_chosen, axis=1)]

    return train_data_df


def create_prog_split_skew_templates_freq_double_phase(domain_train_data_df,
                                                       validation_set_prop=0.05,
                                                       training_size=60000,
                                                       exp=0):
    """
    domain_train_data_df - dataframe with the data for training
    validation_set_prop - proportion of dev set from each training size
    training_size - number of training examples for each experiment
    exp - at each step a template t is sampled with prob p(T=t)**exp, then an example e with template t is sampled uniformly
            when exp=1, this process is equivalent to random sampling
            when exp=0, p(T=t)**exp is uniform hence this process results in more templates compared to random sampling
    """
    training_examples = []

    while len(training_examples) < training_size:
        tmp = domain_train_data_df[~domain_train_data_df.index.isin(training_examples)]
        templs = tmp.groupby('abstract_template')['input'].count().apply(
            lambda x: (x / tmp.shape[0]) ** exp).reset_index()
        t = templs.sample(n=1, weights='input', axis=0)
        e = domain_train_data_df[domain_train_data_df['abstract_template'] == t].sample(n=1)
        training_examples.append(e.index[0])

    sample = domain_train_data_df.filter(items=training_examples, axis=0)

    train_list, dev_list = train_test_split(sample, test_size=validation_set_prop)
    # TODO: save


def sample_sets(training_size,
                set_size,
                train_data_df,
                exmp_id_col,
                weights_temp_col,
                lev_temp_col,
                cache_base_path,
                domain='test'):
    """
    Edit distance: Levenshtein distance. Includes: deletion, replacement, addition.
    Training size s, neighborhood size k, sample S, dataset D
    Compute distance between all level-1 template pairs
    Repeat s//k times:
    * Sample one example e with (uniform dist over all level-4 templates, or over all levels templates), S=S+{e}
    * Retrieve distances between level-1 template of e and all level-1 templates of D-S
    * Sample k nearest level-1 templates, then from each template sample one example and add to S.
    """
    cache_path = Path(cache_base_path) / f"/thingtalk/synth_baseline/{domain}_template_lev_distance.pkl"

    # calculate inverse frequncy of templates p(T=t)^-1, attach to all examples
    if not 'weights' in train_data_df.columns:
        weights = train_data_df[weights_temp_col].value_counts(normalize=True).apply(
            lambda x: x ** (-1)).reset_index().rename(
            columns={weights_temp_col: 'weights', 'index': weights_temp_col})
        train_data_df = train_data_df.merge(weights, on=weights_temp_col, how='left')

    # template1 to <template2 : lev distance> dict
    def load_cache():
        if Path(cache_path).exists():
            with open(cache_path, 'rb') as fp:
                cache = pickle.load(fp)
            return cache
        else:
            return init_cache()

    def init_cache():
        temp_to_id = {temp: i for i, temp in enumerate(train_data_df[lev_temp_col].unique())}
        tempid_to_temp = {v: k for k, v in temp_to_id.items()}
        tempid_to_lev = {tempid: defaultdict() for tempid in tempid_to_temp.keys()}
        return {'temp_to_id': temp_to_id,
                'tempid_to_temp': tempid_to_temp,
                'tempid_to_lev': tempid_to_lev}

    def save_cache(lev_cache):
        with open(cache_path, 'wb') as f:
            pickle.dump(lev_cache, f)

    lev_cache = load_cache()

    sample = []
    sets = int(training_size*0.95//1//(set_size+1))
    train_data_df = train_data_df.sample(frac=1.0)
    for set_id in tqdm(range(sets)):
        example = train_data_df.sample(n=1, weights='weights').iloc[0]
        train_data_df = train_data_df[~(train_data_df[exmp_id_col] == example[exmp_id_col])]

        train_data_df, neighborhood, lev_cache = \
            expand_neighberhood(lev_cache=lev_cache, train_data_df=train_data_df, example=example,
                                exmp_id_col=exmp_id_col, lev_temp_col=lev_temp_col,
                                set_size=set_size, domain=domain)
        assert neighborhood.shape[0] == neighborhood[
            exmp_id_col].nunique(), f"set duplications! {example[exmp_id_col]}, set: {neighborhood[exmp_id_col].unique()}, set_id: {set_id}"
        assert train_data_df.shape[0] == train_data_df[exmp_id_col].nunique(), f"duplications in train! "
        assert neighborhood.shape[
                   0] <= set_size + 1, f"set too big! {example[exmp_id_col]}, set: {neighborhood[exmp_id_col].unique()}, set_id: {set_id}"

        sample.append(neighborhood.assign(set_id=set_id))
        if set_id % 1000 == 0:
            save_cache(lev_cache)

    save_cache(lev_cache)
    sample_df = pd.concat(sample, axis=0)
    dev_df = train_data_df.sample(n=int(training_size * 0.05 // 1)).assign(set_id=-1)
    dev_df = validate_and_fix(dev_df, sample_df['abstract_template'], '', '', prog_col='abstract_template', debug=True)

    # merge train and dev
    split_name = f"{set_size}_{training_size}_0_false"
    sample_df = sample_df.assign(**{split_name: 1})
    dev_df = dev_df.assign(**{split_name: 2})

    return pd.concat([sample_df, dev_df], axis=0)


def expand_neighberhood(lev_cache, train_data_df, example, exmp_id_col,
                        lev_temp_col, set_size, allow_zero=False, domain='test'):
    """
    uses lev_cache to sample the nearest neighbors of example in train_pull.
    returns a pd.DataFrame neighborhood: set_size nearest neighbors + example, and the pd.DataFrame train_pull without
    the neighborhood.
    """
    def lev(t1: str, t2: str, allow_zero=allow_zero):
        t1_list = t1.split()
        t2_list = t2.split()
        d = editdistance.eval(t1_list, t2_list)
        if not allow_zero and d == 0:
            d = 100000
        return d

    one_dist_counter = 0
    # shuffle the order of the examples for the "fast track"
    target_tempid = lev_cache['temp_to_id'][example[lev_temp_col]]
    temps = list(lev_cache['tempid_to_temp'].keys())
    random.shuffle(temps)
    for tempid in temps:
        # check distance from all examples in neighborhood
        lev_val_a = lev_cache['tempid_to_lev'][target_tempid].get(tempid, None)
        lev_val_b = lev_cache['tempid_to_lev'][tempid].get(target_tempid, None)
        if lev_val_a is None and lev_val_b is None:
            lev_cache['tempid_to_lev'][target_tempid][tempid] = \
                lev_cache['tempid_to_lev'][tempid][target_tempid] = \
                                                                   lev(example[lev_temp_col],
                                                                       lev_cache['tempid_to_temp'][tempid],
                                                                        allow_zero=allow_zero)
        elif lev_val_a is None:
            lev_cache['tempid_to_lev'][target_tempid][tempid] = lev_val_b
        elif lev_val_b is None:
            lev_cache['tempid_to_lev'][tempid][target_tempid] = lev_val_a

        one_dist_counter += int(lev_cache['tempid_to_lev'][target_tempid][tempid] == 1)

        # fast track, most examples should have 4 neighbors at distance 1
        if one_dist_counter == set_size:
            break

    # extract the neighbors
    slack = 0
    neighborhood_size = 0
    while neighborhood_size < set_size:
        neighborhood = heapq.nsmallest(set_size+slack, lev_cache['tempid_to_lev'][target_tempid].items(), key=lambda x: x[1])
        neighborhood_temps = [lev_cache['tempid_to_temp'][n[0]] for n in neighborhood]
        neighborhood_size = train_data_df[train_data_df[lev_temp_col].isin(neighborhood_temps)].shape[0]
        slack += 1

    neighborhood_df = train_data_df[train_data_df[lev_temp_col].isin(neighborhood_temps)].sample(n=set_size)

    # update the train_pull
    train_data_df = train_data_df[~(train_data_df[exmp_id_col].isin(neighborhood_df[exmp_id_col].unique()))]

    return train_data_df, neighborhood_df.append(example, ignore_index=True), lev_cache


def create_prog_split_skew_templates_freq_all_levels(train_data_df,
                                                     training_size_list,
                                                     with_para: bool = False,
                                                     validation_set_prop=0.05,
                                                     input_col=1,
                                                     prog_col=2,
                                                     num_splits: int = 5,
                                                     save_space: bool = False):
    """
    domain_train_pull - dataframe with the data for training
    validation_set_prop - proportion of dev set from each training size
    training_size - number of training examples for each experiment
    """
    # fix p(x)
    # calculate the weight of each example as joint probablity of all its tempaltes, where p_l(t=T) is uniform
    N = train_data_df.shape[0]
    uniform_over_templs_top_level = train_data_df.groupby(['template_kbfree_untyped_noops'])[
        'template_kbfree_untyped'].nunique()
    uniform_over_templs_mid_level = train_data_df.groupby(['template_kbfree_untyped'])['template_kbfree'].nunique()
    uniform_over_templs_mid2_level = train_data_df.groupby(['template_kbfree'])['abstract_template'].nunique()
    uniform_over_examples = train_data_df['abstract_template'].value_counts(normalize=False)
    top_level_temps_count = train_data_df['template_kbfree_untyped_noops'].nunique()

    def q(x):
        return (1 / top_level_temps_count) * \
               (1 / uniform_over_templs_top_level[x['template_kbfree_untyped_noops']]) * \
               (1 / uniform_over_templs_mid_level[x['template_kbfree_untyped']]) * \
               (1 / uniform_over_templs_mid2_level[x['template_kbfree']]) * \
               (1 / uniform_over_examples[x['abstract_template']])

    weights = train_data_df.apply(q, axis=1)

    # sample training sets by skewing p(T=t)
    for training_size in training_size_list:
        for i in range(num_splits):
            split_name = f"-2_{training_size}_{i}_{with_para}"
            # sample
            sample = train_data_df.sample(n=training_size, weights=weights)
            tmp_train, tmp_dev = train_test_split(sample, test_size=validation_set_prop)
            # validate
            tmp_dev = validate_and_fix(tmp_dev, tmp_train[prog_col],
                                       f"dev s-{training_size} i-{i}", f"train",
                                       prog_col=prog_col, debug=True)
            # merge
            tmp_train = tmp_train.assign(**{split_name: 1})
            tmp_dev = tmp_dev.assign(**{split_name: 2})
            tmp = pd.concat([tmp_train, tmp_dev], axis=0)
            train_data_df = train_data_df.merge(tmp[[split_name]], left_index=True, right_index=True, how='left')

            assert train_data_df[split_name].value_counts()[1] == tmp_train.shape[0]
            assert train_data_df[split_name].value_counts()[2] == tmp_dev.shape[0]
            assert train_data_df.shape[0] == N

    def is_chosen(x):
        for s in training_size_list:
            for i in range(num_splits):
                split_name = f"-2_{s}_{i}_{with_para}"
                if x[split_name] == 1 or x[split_name] == 2:
                    return True
        return False

    if save_space:
        train_data_df = train_data_df[train_data_df.apply(is_chosen, axis=1)]

    return train_data_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compositional_test_path', type=str,
                        help="absolute path to a csv with compositional test data",
                        default="data/program_test.tsv")
    parser.add_argument('--compositional_dev_path', type=str,
                        help="absolute path to a csv with compositional validation data",
                        default="data/program_dev.tsv")
    parser.add_argument('--augmented_path', type=str, help="absolute path to a csv with question, query pairs",
                        default="data/small_synthetic_data.tsv")
    parser.add_argument('--question_col_index', type=int, default=1,
                        help="index of the questions column in the input csv")
    parser.add_argument('--query_col_index', type=int, default=2,
                        help="index of the (ThingTalk) queries column in the input csv")
    parser.add_argument('--training_size', type=int, default=20,
                        help="training set size to sample")
    parser.add_argument('--save_training_pool', dest='save_training_pool', action='store_true',
                        help="if true, saves the training examples extracted from augmented_path in the same directory")
    parser.add_argument('--create_training_pool', dest='create_training_pool', action='store_true',
                        help="if true, uses the compositional evaluation files to clean the file in augmented_path")
    parser.add_argument('--save_uat_samples', dest='save_uat_samples', action='store_true',
                        help="if true, saves 5 UAT samples in the same directory as augmented_path. the output csv has "
                             "the same columns as the input file, and additional 5 columns, each marks the examples in "
                             "the sample")

    args = parser.parse_args()

    # read question, query pairs
    augmented_data_df = pd.read_csv(args.augmented_path, sep='\t', header=None)
    # convert each query to a template
    augmented_data_df['abstract_template'] = augmented_data_df[args.query_col_index].apply(lambda x: convert_to_schemafree_template_untyped(str(x)))
    augmented_data_df['abstract_template'] = augmented_data_df['abstract_template'].apply(
        convert_to_schemafree_template_untyped_noops)
    # get the compositional development and test templates
    # these templates shouldn't be seen by the model at training
    compositional_data = pd.concat(
        [
            pd.read_csv( args.compositional_test_path, sep='\t', header=None),
            pd.read_csv( args.compositional_dev_path, sep='\t', header=None)
        ]
    )
    test_templs = compositional_data[5].unique()
    if args.create_training_pool:
        # remove <question, query, template> triplets where template \in test_templs
        train_pool_data_df = augmented_data_df[~augmented_data_df['abstract_template'].isin(test_templs)]
    else:
        train_pool_data_df = augmented_data_df

    train_pool_data_df['template'] = train_pool_data_df[args.query_col_index].apply(
        lambda x: convert_to_template(str(x)))

    if args.save_training_pool:
        if not args.create_training_pool:
            print('WARNING: training pull is identical to the input data file')
        train_pool_data_df.to_csv(
            Path(args.augmented_path).parent / "em_preprocessed.tsv",
            sep='\t',
            index=False
        )

    # create UAT splits
    if args.save_uat_samples:
        train_pool_data_df = create_prog_split_skew_templates_freq(train_pool_data_df,
                                                                   [-1],
                                                                   [args.training_size],
                                                                   validation_set_prop=0.20,
                                                                   temp_col='abstract_template',
                                                                   input_col=args.question_col_index,
                                                                   prog_col=args.query_col_index,
                                                                   num_splits=5,
                                                                   save_space=False,
                                                                   with_paraphrase=False)
        train_pool_data_df.to_csv(Path(args.augmented_path).parent / "small_uat_splits.tsv", index=False, sep='\t')
