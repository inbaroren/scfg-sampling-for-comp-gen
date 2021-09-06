import pandas as pd
import re
from unidecode import unidecode
from typing import List, Union
import os


def get_part_path(path, part_idx):
    if path.endswith(os.path.sep):
        has_separator = True
        path = path[:-1]
    else:
        has_separator = False
    return path + '_part' + str(part_idx+1) + (os.path.sep if has_separator else '')


def split_file_on_disk(file_path, num_splits, output_paths=None, delete=False):
    all_output_paths = []
    all_output_files = []
    for part_idx in range(num_splits):
        if output_paths is None:
            output_path = get_part_path(file_path, part_idx)
        else:
            output_path = output_paths[part_idx]
        all_output_paths.append(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_output_files.append(open(output_path, 'w'))

    with open(file_path, 'r') as input_file:
        output_file_idx = 0
        for line in input_file:
            all_output_files[output_file_idx].write(line)
            output_file_idx = (output_file_idx + 1) % len(all_output_files)

    for f in all_output_files:
        f.close()

    if delete:
        os.remove(file_path)

    return all_output_paths


def combine_files_on_disk(file_path_prefix, num_files, line_group_size, delete=False):
    all_input_file_contents = []
    all_input_file_paths = []
    for i in range(num_files):
        input_file_path = get_part_path(file_path_prefix, i)
        all_input_file_paths.append(input_file_path)
        with open(input_file_path, 'r') as f:
            all_input_file_contents.append([line for line in f])

    all_indices = [0] * len(all_input_file_contents)
    finished_reading = [False] * len(all_input_file_contents)
    input_file_idx = 0
    with open(file_path_prefix, 'w') as combined_file:
        while True:
            if finished_reading[input_file_idx]:
                input_file_idx = (input_file_idx + 1) % len(all_input_file_contents)
                continue
            for i in range(line_group_size):
                line = all_input_file_contents[input_file_idx][all_indices[input_file_idx]]
                combined_file.write(line)
                all_indices[input_file_idx] += 1
            if all_indices[input_file_idx] == len(all_input_file_contents[input_file_idx]):
                finished_reading[input_file_idx] = True
                if all(finished_reading):
                    break
            input_file_idx = (input_file_idx + 1) % len(all_input_file_contents)

    if delete:
        for file_path in all_input_file_paths:
            os.remove(file_path)


def down_sampling(df, n: int = 1000, template_col: Union[str, int] = 'template', query_col: Union[str, int] = 1):
    """
    Sample up to n examples from each program template
    df - pd.DataFrame with column "template" with the program templates
    n - max number of examples for program template. default 1k
    """
    new = []

    # choose only the templates that are too common
    templates_count = df.groupby(template_col)[query_col].count()
    new.append(df[df[template_col].isin(templates_count[templates_count <= n].index)])
    common_templates = templates_count[templates_count > n].reset_index(drop=False)[template_col].values

    for t in common_templates:
        s = df[df[template_col] == t].sample(n=min(df[df[template_col] == t].shape[0], n), random_state=42)
        new.append(s)
    new_df = pd.concat(new, axis=0)

    return new_df


def convert_to_template(x: str, debug=False):
    """
        Convert a ThingTalk program to a template by annonymizing the quoted strings and variables
        The types of the variables:
            boolean
            string (quoted)
            number
            enum
            location
        :param x: str ThinkTalk program
        :return: str
    """
    # annonymize all values (and functions of values) to 'VALUE'
    i = 1
    x = unidecode(x)
    template = re.sub(r' true ', ' BOOL_VAL ', str(x))
    if debug:  # 1
        print(i, template)
        i += 1
    template = re.sub(r' false ', ' BOOL_VAL ', template)
    if debug:  # 2
        print(i, template)
        i += 1
    template = re.sub(r'" [0-9A-Za-z\'_\-.:;,?\s\\\/&@()+]+ "', 'QUOTED_VAL', template)
    if debug:  # 3
        print(i, template)
        i += 1
    template = re.sub(r' (NUMBER_[0-9]|[0-9]{1,2}) unit:([a-z]+) ', ' NUMBER_VAL ', template)
    if debug:  # 4
        print(i, template)
        i += 1
    template = re.sub(r' (end_of|start_of) unit:([a-z]+) ', ' TIME_VAL ', template)
    if debug:  # 5
        print(i, template)
        i += 1

    template = re.sub(r'enum:[a-zA-Z_:.0-9]+', 'ENUM_VAL', template)
    if debug:  # 6
        print(i, template)
        i += 1
    template = re.sub(r'new Date \( [a-zA-Z_0-9\s,]+ \)', 'DATE_VAL', template)
    if debug:  # 7
        print(i, template)
        i += 1
    template = re.sub(r'time:[0-9]{1,2}:0:0', 'TIME_VAL', template)
    if debug:  # 8
        print(i, template)
        i += 1
    template = re.sub(r' now ', ' TIME_VAL ', template)
    template = re.sub(r'[ ]+', ' ', template)
    if debug:  # 9
        print(i, template)
        i += 1

    template = re.sub(r'(==|,) location:\s{0,1}[a-zA-Z_:.0-9]+ ', r'\g<1> LOCATION_VAL ', template)
    if debug:  # 10
        print(i, template)
        i += 1
    template = re.sub(r' [0-9]{1,2} ', ' NUMBER_VAL ', template)
    if debug:  # 11
        print(i, template)
        i += 1

    template = re.sub(r'EMAIL_ADDRESS_[0-9]', 'QUOTED_VAL', template)
    template = re.sub(r'PHONE_NUMBER_[0-9]', 'QUOTED_VAL', template)
    template = re.sub(r'QUOTED_STRING_[0-9]', 'QUOTED_VAL', template)
    template = re.sub(r'NUMBER_[0-9]', 'NUMBER_VAL', template)
    template = re.sub(r'([A-Z]+)_[0-9]', '\g<1>_VAL', template)
    template = re.sub(r'location:VAL', 'LOCATION_VAL', template)
    template = re.sub(r'TIME_VAL', 'NUMBER_VAL', template)
    template = re.sub(r'DATE_VAL', 'NUMBER_VAL', template)

    if debug:  # 11
        print(i, template)
        i += 1

    template = re.sub(r'[ ]+', ' ', template)
    return template


def convert_to_schemafree_template(template: str):
    """
        Convert a ThingTalk program to a template by annonymizing the quoted strings and variables, AND annonymizing
        all the properties
        :param template: str ThinkTalk template (program with annonimyzed values)
        :return: str
    """

    # remove schema properties and tables
    template = re.sub(r'org.schema.[a-zA-Z_]+:[a-zA-Z_]+', 'schema_property', template)
    template = re.sub(r'@org.schema.[a-zA-Z_]+\.[a-zA-Z_]+', 'schema_table', template)

    # remove ^^property (denotes the kb relation / type of the value)
    template = re.sub(r'tt:[a-z_]+', 'tt_property', template)
    template = re.sub(r'\^\^tt_property', '', template)
    template = re.sub(r'\^\^schema_property', '', template)

    # remove other patameters
    template = re.sub(r' param:[a-zA-Z\._]+[:\.]([a-zA-Z\._]+)(\([a-zA-Z\.:@_,]+\))?',
                      r' param:property', template)
    template = re.sub(r' param:[a-zA-Z\._]+ ', r' param:property ', template)
    template = template.replace('param:property(Entity(schema_property))',
                                'param:property')
    template = template.replace('param:property(Entity(tt_property))',
                                'param:property')

    #     for enum_prop in ['param:property:Enum(AudiobookFormat,EBook,GraphicNovel,Hardcover,Paperback)',
    #  'param:property:Enum(cheap,moderate,expensive,luxury)']:
    #         template = template.replace(enum_prop, 'param:property:Enum')

    template = re.sub(r'[ ]+', ' ', template)
    return template


def get_file_as_df(f, remove_index=True, drop_dups=True):
    question_col = 1
    with open(f) as fp:
        lines = fp.readlines()
    clean_lines = [l.strip('\n').split('\t') for l in lines]
    df = pd.DataFrame(clean_lines)
    if df.shape[0] == 0:
        return None
    assert df.shape[0] == len(clean_lines)
    # clean the columns as specified
    if not df.shape[1] == 2 and remove_index:
        df = df[[1,2]]
        df.columns = [0,1]
        question_col = 0
    elif df.shape[1] == 2:
        df.columns = [0, 1]
        question_col = 1
    # drop duplications (can't have two examples with the same input question
    if drop_dups:
        df = df.drop_duplicates(question_col)

    return df


def get_properties_set(templates):
    properties = set()
    for template in templates:
        tt_prop = re.findall(r'(?:[\s(]{0,1})tt:[a-z_]+(?:[\s)])', template)
        schema_property = re.findall(r'(?:[\s(]{0,1})org.schema.[a-zA-Z_]+:[a-zA-Z_]+(?:[\s)])', template)
        schema_table = re.findall(r'(?:[\s(]{0,1})@org.schema.[a-zA-Z_]+\.[a-zA-Z_]+(?:[\s)])', template)
        property_typed = re.findall(r'(?:[\s(]{0,1})param:[a-zA-Z._]+[:.][a-zA-Z._]+\([()a-zA-Z.:@_,]+\)(?:[\s)])',
                                    template)
        property = re.findall(r'(?:[\s(]{0,1})param:[a-zA-Z._]+[:.][a-zA-Z._]+(?:[\s)])', template)
        property2 = re.findall(r'(?:[\s(]{0,1})param:[a-zA-Z]+(?:[\s)])', template)
        enums = re.findall(r'enum:[a-zA-Z_:.0-9]+', template)
        properties.update(tt_prop + schema_property + schema_table + property + property_typed + enums + property2)
    return properties


def get_consts(templates, consts):
    consts_in_temps = set()
    for const in consts:
        if not templates[templates.apply(lambda x: f" {const} " in x)].empty:
            consts_in_temps.add(const)
    return consts_in_temps


def validate_split(test_templates, train_templates, debug=True):
    # verfiy all kb properties are in test are subset of train
    test_properties = get_properties_set(test_templates)
    train_properties = get_properties_set(train_templates)
    properties_are_subset = test_properties.issubset(train_properties)

    if debug and not properties_are_subset:
        print("properties_are_subset is False, example: ", len(test_properties.difference(train_properties)))
        if len(test_properties.difference(train_properties)) < 10:
            print(test_properties.difference(train_properties))

    # verfiy all language constants are in test are subset of train
    consts = ["-", "+", "==", "<=", ">=", "=~", "~=", "contains~", "contains", "in_array~", "in_array",
              "and", "or", "not", "compute", "count", "sort", "distance", "aggregate", "avg", "sum", "asc", "desc"]

    consts_in_test = get_consts(test_templates, consts)
    const_in_train = get_consts(train_templates, consts)
    consts_subset = consts_in_test.issubset(const_in_train)

    if debug and not consts_subset:
        print("consts_are_subset is False, example: ",
              len(consts_in_test.difference(const_in_train)),
              consts_in_test.difference(const_in_train))

    return properties_are_subset and consts_subset, test_properties.difference(
        train_properties), consts_in_test.difference(const_in_train)


def remove_examples_by_program(df, condition_string, prog_col=2):
    condition_string = f" {condition_string.strip()} "
    orig_size = df.shape[0]
    df = df[~df[prog_col].apply(lambda x: condition_string in x)]
    print(f"Lost {orig_size-df.shape[0]}/{orig_size} rows after removing examples with {condition_string}")
    return df


def validate_and_fix(test_df, train_templates, test_name, train_name, prog_col=2, debug=True):
    if debug: print(f"\n\nValidate and fix test: {test_name}, train: {train_name}")
    valid, rmv_props, rmv_consts = validate_split(test_df[prog_col], train_templates, debug=debug)
    for prop in rmv_props.union(rmv_consts):
        test_df = remove_examples_by_program(test_df, prop, prog_col=prog_col)
    return test_df


def find_larger_index(x, substring1, substring2):
    """
    x.index("substring1") < x.index("substring2")
    """
    try:
        index1 = x.index(substring1)
    except ValueError:
        return False
    try:
        index2 = x.index(substring2)
    except ValueError:
        return False
    return index1 < index2


def crowdsourced_properties_fix(x, domain):
    fixes = {"books": {'param:aggregateRating.ratingCount': 'param:aggregateRating.ratingCount:Number',
                       'param:aggregateRating.ratingValue': 'param:aggregateRating.ratingValue:Number',
                       'param:id:String': "param:id:Entity(org.schema.Book:Book)",
                       'param:bookFormat:Enum(EBook,AudiobookFormat,Hardcover,Paperback)': "param:bookFormat:Enum(AudiobookFormat,EBook,GraphicNovel,Hardcover,Paperback)"},
             "movies": {
                 'param:aggregateRating.ratingValue': 'param:aggregateRating.ratingValue:Number'
             },
             "restaurants": {
                 'param:aggregateRating.reviewCount': 'param:aggregateRating.reviewCount:Number',
                 'param:aggregateRating.ratingValue': 'param:aggregateRating.ratingValue:Number',
                 'param:priceRange:String': 'param:priceRange:Enum'
             }
             }
    return ' '.join([fixes.get(domain, {}).get(tok, tok) for tok in x.split()])


def paraphrased_properties_fix(x, domain):
    fixes = {
                "books": {
                    'param:bookFormat:Enum(EBook,AudiobookFormat,Hardcover,Paperback)': "param:bookFormat:Enum(AudiobookFormat,EBook,GraphicNovel,Hardcover,Paperback)"
                },
                "restaurants": {
                    'param:priceRange:String': 'param:priceRange:Enum'
                }
             }
    return ' '.join([fixes.get(domain, {}).get(tok, tok) for tok in x.split()])


def convert_to_schemafree_template_untyped(program_str: str, debug=False):
    """
        Convert a ThingTalk program to a template by annonymizing the quoted strings and variables, AND annonymizing
        all the properties
        :param program_str: str ThinkTalk program
        :return: str
    """
    # annonymize all values (and functions of values) to 'VALUE'
    i = 1
    program_str = unidecode(program_str)
    template = re.sub(r' true ', ' VALUE ', str(program_str))
    if debug:  # 1
        print(i, template)
        i += 1
    template = re.sub(r' false ', ' VALUE ', template)
    if debug:  # 2
        print(i, template)
        i += 1
    template = re.sub(r'" [0-9A-Za-z\'_\-.:;,?\s\\\/&@()+]+ "', 'VALUE', template)
    if debug:  # 3
        print(i, template)
        i += 1
    template = re.sub(r' (NUMBER_[0-9]|start_of|end_of|[0-9]{1,2}) unit:([a-z]+) ', ' NUMBER_VAL ', template)
    if debug:  # 4
        print(i, template)
        i += 1
    template = re.sub(r' [0-9]{1,2} ', ' NUMBER_VAL ', template)
    if debug:  # 5
        print(i, template)
        i += 1
    template = re.sub(r'enum:[a-zA-Z_:.0-9]+', 'ENUM_VAL', template)
    if debug:  # 6
        print(i, template)
        i += 1
    template = re.sub(r'new Date \( [a-zA-Z_0-9\s,]+ \)', 'NUMBER_VAL', template)
    if debug:  # 7
        print(i, template)
        i += 1
    template = re.sub(r'time:[0-9]{1,2}:0:0', 'NUMBER_VAL', template)
    if debug:  # 8
        print(i, template)
        i += 1
    template = re.sub(r' now ', ' NUMBER_VAL ', template)
    template = re.sub(r'[ ]+', ' ', template)
    if debug:  # 9
        print(i, template)
        i += 1

    template = re.sub(r'(==|,) location:\s{,1}[a-zA-Z_:.0-9]+ ', r'\g<1> VALUE ', template)
    if debug:  # 10
        print(i, template)
        i += 1
    template = re.sub(
        r'(EMAIL_ADDRESS_[0-9]|PHONE_NUMBER_[0-9]|QUOTED_STRING_[0-9]|QUOTED_VAL|UNIT_VAL|NUMBER_VAL|location:VAL|ENUM_VAL|NUMBER_0|[A-Z]+_[0-9])',
        'VALUE', template)
    if debug:  # 11
        print(i, template)
        i += 1
    template = re.sub(r'\[\s+VALUE\s+,\s+VALUE\s+\]', 'VALUE', template)
    if debug:  # 12
        print(i, template)
        i += 1
    template = re.sub(r' \( VALUE \)[\s]+(-|\+)[\s]+VALUE ', ' VALUE ', template)
    if debug:  # 13
        print(i, template)
        i += 1

    # replace all functions of VALUE / NUMBER_VAL
    for _ in range(5):
        numval_matches = re.findall(
            r"(\({0,1}[\s]+(?:NUMBER_VAL|VALUE)[\s]+(?:-|\+)[\s]+(?:NUMBER_VAL|VALUE)[\s]+\){0,1})", template)
        for m in numval_matches:
            m_string = m
            if len(re.findall('\(', m_string)) == len(re.findall('\)', m_string)):
                template = template.replace(m_string, ' VALUE ')
            else:
                template = template.replace(m_string.strip('()'), ' VALUE ')
        if debug:
            print(i, template)
            i += 1

    # remove schema properties and tables
    template = re.sub(r'org.schema.[a-zA-Z_]+:[a-zA-Z_]+', 'schema_property', template)
    template = re.sub(r'@org.schema.[a-zA-Z_]+\.[a-zA-Z_]+', 'schema_table', template)

    # remove ^^property (denotes the kb relation / type of the value)
    template = re.sub(r'tt:[a-z_]+', 'tt_property', template)
    template = re.sub(r'\^\^tt_property', '', template)
    template = re.sub(r'\^\^schema_property', '', template)

    # remove other patameters
    template = re.sub(r' param:[a-zA-Z\._]+[:\.]([a-zA-Z\._]+)(\([a-zA-Z\.:@_,]+\))?',
                      r' param:property', template)
    template = re.sub(r' param:[a-zA-Z\._]+ ', r' param:property ', template)
    template = template.replace('param:property(Entity(schema_property))',
                                'param:property')
    template = template.replace('param:property(Entity(tt_property))',
                                'param:property')

    template = re.sub(r'[ ]+', ' ', template)
    return template


def convert_to_schemafree_template_untyped_noops(kbfree_untyped_templ):
    for op in ("==", "<=", ">=", "=~", "~=", "contains~", "contains", "in_array~", "in_array"):
        kbfree_untyped_templ = kbfree_untyped_templ.replace(f" {op} ", ' REL_OP ')
    for op in ("and", "or", "not"):
        kbfree_untyped_templ = kbfree_untyped_templ.replace(f" {op} ", ' LOG_OP ')
    for op in ("compute", "count", "sort", "distance", "aggregate", "avg", "sum"):
        kbfree_untyped_templ = kbfree_untyped_templ.replace(f" {op} ", ' FUNC_OP ')
    for op in ("asc", "desc"):
        kbfree_untyped_templ = kbfree_untyped_templ.replace(f" {op} ", ' FUNC_MOD ')

    kbfree_untyped_templ = re.sub('FUNC_OP FUNC_OP', 'FUNC_OP', kbfree_untyped_templ)
    kbfree_untyped_templ = re.sub('[ ]+', ' ', kbfree_untyped_templ)

    return kbfree_untyped_templ