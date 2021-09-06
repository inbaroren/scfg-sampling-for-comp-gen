import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import editdistance
import json
from collections import defaultdict, namedtuple
import random
import heapq
from scripts.utils import validate_and_fix
from tqdm import tqdm
import pickle


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


def create_prog_split_skew_templates_freq(train_pull,
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
    domain_train_pull - dataframe with the data for training
    validation_set_prop - proportion of dev set from each training size
    training_size - number of training examples for each experiment
    """
    # calculate frequncy of templates p(T=t), attach to all examples
    N = train_pull.shape[0]
    freq = train_pull[temp_col].value_counts(normalize=True).reset_index().rename(
        columns={temp_col: 'freq', 'index': temp_col})
    train_pull = train_pull.merge(freq, on=temp_col, how='left')

    # sample training sets by skewing p(T=t)
    for training_size in training_size_list:
        for exp in exp_list:
            for i in range(num_splits):
                split_name = f"{exp}_{training_size}_{i}_{with_paraphrase}".lower()
                # fix prob distribution
                weights = train_pull['freq'].apply(lambda x: x ** exp)
                # sample
                sample = train_pull.sample(n=training_size, weights=weights)
                tmp_train, tmp_dev = train_test_split(sample, test_size=validation_set_prop)
                # validate
                tmp_dev = validate_and_fix(tmp_dev, tmp_train[prog_col],
                                           f"dev exp-{exp} s-{training_size} i-{i}", f"train",
                                           prog_col=prog_col, debug=True)
                # merge
                tmp_train = tmp_train.assign(**{split_name: 1})
                tmp_dev = tmp_dev.assign(**{split_name: 2})
                tmp = pd.concat([tmp_train, tmp_dev], axis=0)
                train_pull = train_pull.merge(tmp[[split_name]], left_index=True, right_index=True, how='left')

                assert train_pull[split_name].value_counts()[1] == tmp_train.shape[0]
                assert train_pull[split_name].value_counts()[2] == tmp_dev.shape[0]
                assert train_pull.shape[0] == N

    def is_chosen(x):
        for s in training_size_list:
            for exp in exp_list:
                for i in range(num_splits):
                    split_name = f"{exp}_{s}_{i}_{with_paraphrase}"
                    if x[split_name] == 1 or x[split_name] == 2:
                        return True
        return False

    if save_space:
        train_pull = train_pull[train_pull.apply(is_chosen, axis=1)]

    return train_pull


def create_prog_split_skew_templates_freq_double_phase(domain_train_pull,
                                                       validation_set_prop=0.05,
                                                       training_size=60000,
                                                       exp=0):
    """
    domain_train_pull - dataframe with the data for training
    validation_set_prop - proportion of dev set from each training size
    training_size - number of training examples for each experiment
    exp - at each step a template t is sampled with prob p(T=t)**exp, then an example e with template t is sampled uniformly
            when exp=1, this process is equivalent to random sampling
            when exp=0, p(T=t)**exp is uniform hence this process results in more templates compared to random sampling
    """
    training_examples = []

    while len(training_examples) < training_size:
        tmp = domain_train_pull[~domain_train_pull.index.isin(training_examples)]
        templs = tmp.groupby('template')['input'].count().apply(
            lambda x: (x / tmp.shape[0]) ** exp).reset_index()
        t = templs.sample(n=1, weights='input', axis=0)
        e = domain_train_pull[domain_train_pull['template'] == t].sample(n=1)
        training_examples.append(e.index[0])

    sample = domain_train_pull.filter(items=training_examples, axis=0)

    train_list, dev_list = train_test_split(sample, test_size=validation_set_prop)
    # TODO: save


def sample_sets(training_size,
                set_size,
                train_pull,
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
    if not 'weights' in train_pull.columns:
        weights = train_pull[weights_temp_col].value_counts(normalize=True).apply(
            lambda x: x ** (-1)).reset_index().rename(
            columns={weights_temp_col: 'weights', 'index': weights_temp_col})
        train_pull = train_pull.merge(weights, on=weights_temp_col, how='left')

    # template1 to <template2 : lev distance> dict
    def load_cache():
        if Path(cache_path).exists():
            with open(cache_path, 'rb') as fp:
                cache = pickle.load(fp)
            return cache
        else:
            return init_cache()

    def init_cache():
        temp_to_id = {temp: i for i, temp in enumerate(train_pull[lev_temp_col].unique())}
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
    train_pull = train_pull.sample(frac=1.0)
    for set_id in tqdm(range(sets)):
        example = train_pull.sample(n=1, weights='weights').iloc[0]
        train_pull = train_pull[~(train_pull[exmp_id_col] == example[exmp_id_col])]

        train_pull, neighborhood, lev_cache = \
            expand_neighberhood(lev_cache=lev_cache, train_pull=train_pull, example=example,
                                exmp_id_col=exmp_id_col, lev_temp_col=lev_temp_col,
                                set_size=set_size, domain=domain)
        assert neighborhood.shape[0] == neighborhood[
            exmp_id_col].nunique(), f"set duplications! {example[exmp_id_col]}, set: {neighborhood[exmp_id_col].unique()}, set_id: {set_id}"
        assert train_pull.shape[0] == train_pull[exmp_id_col].nunique(), f"duplications in train! "
        assert neighborhood.shape[
                   0] <= set_size + 1, f"set too big! {example[exmp_id_col]}, set: {neighborhood[exmp_id_col].unique()}, set_id: {set_id}"

        sample.append(neighborhood.assign(set_id=set_id))
        if set_id % 1000 == 0:
            save_cache(lev_cache)

    save_cache(lev_cache)
    sample_df = pd.concat(sample, axis=0)
    dev_df = train_pull.sample(n=int(training_size*0.05//1)).assign(set_id=-1)
    dev_df = validate_and_fix(dev_df, sample_df['template'], '', '', prog_col='template', debug=True)

    # merge train and dev
    split_name = f"{set_size}_{training_size}_0_false"
    sample_df = sample_df.assign(**{split_name: 1})
    dev_df = dev_df.assign(**{split_name: 2})

    return pd.concat([sample_df, dev_df], axis=0)


def expand_neighberhood(lev_cache, train_pull, example, exmp_id_col,
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
        neighborhood_size = train_pull[train_pull[lev_temp_col].isin(neighborhood_temps)].shape[0]
        slack += 1

    neighborhood_df = train_pull[train_pull[lev_temp_col].isin(neighborhood_temps)].sample(n=set_size)

    # update the train_pull
    train_pull = train_pull[~(train_pull[exmp_id_col].isin(neighborhood_df[exmp_id_col].unique()))]

    return train_pull, neighborhood_df.append(example, ignore_index=True), lev_cache


def create_prog_split_skew_templates_freq_all_levels(train_pull,
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
    N = train_pull.shape[0]
    uniform_over_templs_top_level = train_pull.groupby(['template_kbfree_untyped_noops'])[
        'template_kbfree_untyped'].nunique()
    uniform_over_templs_mid_level = train_pull.groupby(['template_kbfree_untyped'])['template_kbfree'].nunique()
    uniform_over_templs_mid2_level = train_pull.groupby(['template_kbfree'])['template'].nunique()
    uniform_over_examples = train_pull['template'].value_counts(normalize=False)
    top_level_temps_count = train_pull['template_kbfree_untyped_noops'].nunique()

    def q(x):
        return (1 / top_level_temps_count) * \
               (1 / uniform_over_templs_top_level[x['template_kbfree_untyped_noops']]) * \
               (1 / uniform_over_templs_mid_level[x['template_kbfree_untyped']]) * \
               (1 / uniform_over_templs_mid2_level[x['template_kbfree']]) * \
               (1 / uniform_over_examples[x['template']])

    weights = train_pull.apply(q, axis=1)

    # sample training sets by skewing p(T=t)
    for training_size in training_size_list:
        for i in range(num_splits):
            split_name = f"-2_{training_size}_{i}_{with_para}"
            # sample
            sample = train_pull.sample(n=training_size, weights=weights)
            tmp_train, tmp_dev = train_test_split(sample, test_size=validation_set_prop)
            # validate
            tmp_dev = validate_and_fix(tmp_dev, tmp_train[prog_col],
                                       f"dev s-{training_size} i-{i}", f"train",
                                       prog_col=prog_col, debug=True)
            # merge
            tmp_train = tmp_train.assign(**{split_name: 1})
            tmp_dev = tmp_dev.assign(**{split_name: 2})
            tmp = pd.concat([tmp_train, tmp_dev], axis=0)
            train_pull = train_pull.merge(tmp[[split_name]], left_index=True, right_index=True, how='left')

            assert train_pull[split_name].value_counts()[1] == tmp_train.shape[0]
            assert train_pull[split_name].value_counts()[2] == tmp_dev.shape[0]
            assert train_pull.shape[0] == N

    def is_chosen(x):
        for s in training_size_list:
            for i in range(num_splits):
                split_name = f"-2_{s}_{i}_{with_para}"
                if x[split_name] == 1 or x[split_name] == 2:
                    return True
        return False

    if save_space:
        train_pull = train_pull[train_pull.apply(is_chosen, axis=1)]

    return train_pull
